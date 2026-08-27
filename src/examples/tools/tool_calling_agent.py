"""Tool Use example using AWS Bedrock and LangChain.

Demonstrates the Tool Use (a.k.a. function calling) agentic pattern: the LLM
is handed a set of tools it may call, and on each turn it either asks for one
or more tool calls or produces a final answer. A driver loop executes the
tools the model requests, feeds their results back as ToolMessages, and
re-invokes the model until it stops asking for tools (or a step cap is hit).

Unlike the earlier examples, the control flow here isn't a fixed LCEL
pipeline — the *model* decides at runtime which tools to call and in what
order, so the loop has to be an actual loop rather than a `prompt | llm`
chain.

How this differs from the other patterns in this repo:
  - prompt_chaining: a fixed two-step `|` chain; every run does the same
    steps in the same order.
  - routing: the LLM classifies once, then deterministic Python dispatches.
  - parallelization: a fixed fan-out of independent sub-chains.
  - reflection: a hand-written loop, but the *sequence* of steps is fixed;
    only the iteration count is dynamic.
  - tools (this file): both the choice of step and the number of steps are
    decided by the model at runtime, so we need the observe -> decide ->
    act -> repeat loop implemented in `run_agent()`.

Message flow for one query (the list re-sent to the model each turn):

    SystemMessage  ── static instructions
    HumanMessage   ── the user's query
    AIMessage      ── model turn 1: carries `.tool_calls` (a request to run tools)
    ToolMessage(s) ── our results for turn 1's calls, one per tool_call_id
    AIMessage      ── model turn 2: either more `.tool_calls` or a final answer
    ...            ── repeats until an AIMessage has no tool_calls, or MAX_STEPS
"""

import logging

# LangChain's message classes. Each maps to a "role" the chat model
# understands:
#   SystemMessage  - developer instructions that frame the whole conversation
#   HumanMessage   - input from the user
#   AIMessage      - a turn produced by the model; may carry `.tool_calls`
#   ToolMessage    - the result of a tool call we ran, fed back to the model
#   BaseMessage    - the common superclass, used only for the list type hint
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# `@tool` wraps a plain Python function so LangChain can expose it to the
# model as a callable tool with a JSON schema derived from the signature.
from langchain_core.tools import tool

# Shared ChatBedrock instance, constructed once at import time in
# src/app/bedrock.py (which also validates AWS credentials at module scope).
from ...app.bedrock import llm

logger = logging.getLogger(__name__)

# The agent loop is: call model -> run any tools it asked for -> call model
# again with the results. This caps how many times that can repeat, so a
# model that keeps requesting tools (or ping-pongs between two of them) can't
# spin forever. Each unit of MAX_STEPS is one model invocation; the tools
# requested within a step don't count individually.
MAX_STEPS: int = 5

# Sample queries for testing. Each is picked to exercise the pattern
# differently:
#   1. A single arithmetic call -> one `calculator` invocation, then answer.
#   2. Two `get_weather` calls (Tokyo, Paris) in one turn, then the model
#      does the "warmer?" comparison itself from the two results.
#   3. A query the model should answer with one `word_count` call.
SAMPLE_QUERIES: list[str] = [
    "What is 24 * 7 + 15?",
    "What's the weather in Tokyo, and is it warmer there than in Paris?",
    "How many words are in 'the quick brown fox jumps over the lazy dog'?",
]


# --- Tools -------------------------------------------------------------------
# The @tool decorator turns a plain function into a LangChain tool: its name,
# docstring, and type-hinted signature become the JSON schema the model sees.
# The docstring is not a comment here — it's the only description the model
# gets of what the tool does and when to use it, so it has to be written for
# that audience. Likewise the parameter names (`expression`, `city`, `text`)
# and their type hints are sent to the model verbatim, so they should read
# clearly on their own.
#
# Every tool here returns a `str`. That keeps the value trivially
# serialisable into a ToolMessage; a real tool can return richer data as long
# as it can be turned into text the model can read.


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as "3 * (4 + 5)" and return the result."""
    # Never eval() raw model output as-is. Restrict the input to arithmetic
    # characters first, then eval with no builtins and no names in scope, so
    # the worst a bad expression can do is raise.
    #
    # `allowed` is the whitelist of characters an arithmetic expression may
    # contain: digits, the four operators, parentheses, decimal point, space.
    allowed = set("0123456789+-*/(). ")
    # Reject empty input, or any expression containing a character outside the
    # whitelist (set difference is non-empty). This blocks names, attribute
    # access, calls, etc. before they ever reach eval().
    if not expression or set(expression) - allowed:
        return f"Error: unsupported characters in expression {expression!r}"
    try:
        # `{"__builtins__": {}}` as globals removes access to built-in
        # functions; `{}` as locals means no names are in scope. Combined
        # with the character whitelist above, this is a locked-down eval.
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - whitelisted above
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError) as exc:
        # Return the error as a normal string rather than raising: the model
        # sees the failure in the ToolMessage and can recover (e.g. retry
        # with a corrected expression) instead of the whole run crashing.
        return f"Error: could not evaluate {expression!r} ({exc})"
    return str(result)


# Stand-in for a real weather API — kept offline so the example is
# self-contained and deterministic. Keys are lower-cased city names so the
# lookup in get_weather() can normalise input to match.
_WEATHER_DB: dict[str, str] = {
    "tokyo": "22 degrees C, clear",
    "paris": "16 degrees C, light rain",
    "new york": "12 degrees C, windy",
    "london": "14 degrees C, overcast",
}


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city. Returns temperature and sky conditions."""
    # Normalise the model-supplied city name (trim whitespace, lower-case)
    # before looking it up, since the model may capitalise or pad it.
    report = _WEATHER_DB.get(city.strip().lower())
    if report is None:
        # Unknown city: instead of a bare failure, tell the model which
        # cities *are* available so it can correct itself on the next turn.
        known = ", ".join(sorted(_WEATHER_DB))
        return f"No weather data for {city!r}. Known cities: {known}."
    return report


@tool
def word_count(text: str) -> str:
    """Count the number of whitespace-separated words in the given text."""
    # str.split() with no argument splits on arbitrary runs of whitespace and
    # discards empty strings, so leading/trailing/repeated spaces don't skew
    # the count. Return as a string to match the other tools' contract.
    return str(len(text.split()))


# Single source of truth for the tool set. TOOLS is what gets bound to the
# model; TOOLS_BY_NAME is how the loop looks a tool up when the model asks
# for it by name. Keeping both derived from one list avoids the two drifting
# apart when a tool is added or removed.
TOOLS = [calculator, get_weather, word_count]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_agent(query: str) -> str:
    """Run the tool-calling loop for a single query and return the final answer.

    The conversation is just a growing list of messages: the model's
    AIMessages (which may carry `tool_calls`) and our ToolMessages (which
    carry the results). Each iteration re-sends the whole list, which is how
    the model "remembers" what it already looked up.
    """
    # bind_tools attaches the tool schemas to the model so it can emit
    # structured tool calls. It doesn't execute anything — running the tools
    # is entirely our job below. The return value is a new runnable; the
    # original `llm` is left untouched.
    model_with_tools = llm.bind_tools(TOOLS)

    # Seed the conversation. This list is mutated in place throughout the
    # loop: we append the model's AIMessage and our ToolMessages each turn,
    # then re-send the whole thing.
    messages: list[BaseMessage] = [
        SystemMessage(
            "You are a helpful assistant. Use the provided tools when they help "
            "answer the question, and you may call more than one. Once you have "
            "enough information, reply to the user directly without calling a tool."
        ),
        HumanMessage(query),
    ]

    # Bounded loop: at most MAX_STEPS model invocations. `step` is 1-based
    # purely so the log messages read naturally.
    for step in range(1, MAX_STEPS + 1):
        # One model turn over the full running transcript. The model sees
        # every prior tool result here, which is what lets it chain calls or
        # summarise at the end.
        ai_message: AIMessage = model_with_tools.invoke(messages)
        messages.append(ai_message)

        # No tool calls on this turn => the model has produced its final
        # answer and the loop is done. `.content` is the user-facing text.
        if not ai_message.tool_calls:
            logger.info("Agent finished after %d step(s)", step)
            return str(ai_message.content)

        # Otherwise, run every tool the model asked for this turn. A single
        # AIMessage can contain multiple tool_calls (e.g. Tokyo *and* Paris
        # weather), so this is a loop. Each `call` is a dict with keys
        # "name", "args", and "id".
        for call in ai_message.tool_calls:
            # Look the tool up by the name the model used. If the model
            # hallucinates a tool that doesn't exist, don't crash — report
            # it back as an error string so the model can adjust.
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                output = f"Error: unknown tool {call['name']!r}"
            else:
                logger.info("Calling tool %s with args %s", call["name"], call["args"])
                # `.invoke(call["args"])` runs the underlying function with
                # the model-supplied kwargs (e.g. {"expression": "24*7+15"}).
                output = tool_fn.invoke(call["args"])
            # Echo each call and its result to stdout so the pattern is
            # visible when running the example.
            print(f"  [tool] {call['name']}({call['args']}) -> {output}")
            # Feed the result back as a ToolMessage tagged with the matching
            # tool_call_id. The id is how the model lines each result up with
            # the specific call it made — essential when several tools ran in
            # the same turn.
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    # Fell out of the for-loop without hitting the `return` above: the model
    # kept asking for tools until the step cap. Return a sentinel string
    # rather than raising so the caller (handle_requests) can keep going with
    # the remaining sample queries.
    logger.warning("Agent hit MAX_STEPS (%d) without a final answer", MAX_STEPS)
    return "Stopped: reached the maximum number of tool-calling steps."


def handle_requests() -> None:
    """Main entry point: run the tool-using agent over the sample queries.

    This is the function registered in the CLI (see src/examples/__main__.py
    `_run_tools` and src/app/cli.py `CMD_TOOLS`), invoked by:
        uv run python -m src.examples tools
    """
    logger.info("Building tool-using agent with %d tools", len(TOOLS))
    print("Running tool-use agent over sample queries")

    # Each query is an independent run: run_agent() starts a fresh message
    # list every time, so nothing carries over between queries.
    for query in SAMPLE_QUERIES:
        print("=" * 60)
        print(f"Query: {query}")
        answer = run_agent(query)
        print("-" * 60)
        print(f"Answer: {answer}")
        print("=" * 60)
        print()
