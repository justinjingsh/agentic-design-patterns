"""Planning example using AWS Bedrock and LangChain.

Demonstrates the Planning (a.k.a. plan-and-execute) agentic pattern: instead
of answering a goal in one shot, the model first writes an explicit,
ordered plan — a list of concrete sub-steps — and a driver then executes
those steps one at a time, feeding the results of the finished steps back in
as context for the next one. A final pass synthesises the per-step results
into the answer to the original goal.

Splitting "decide what to do" from "do it" is the whole point: the plan is
produced once, up front, and is visible and inspectable before any step
runs; execution is then a plain deterministic loop over that plan.

How this differs from the other patterns in this repo:
  - prompt_chaining: a fixed two-step `|` chain — the steps are hard-coded by
    the programmer, not written by the model.
  - routing: the model makes one classification decision, then deterministic
    Python dispatches; there is no multi-step plan.
  - parallelization: a fixed fan-out of independent sub-tasks with no
    ordering between them.
  - reflection: a loop that improves a single draft; the sequence of steps is
    fixed, only the iteration count is dynamic.
  - tools: the model interleaves planning and acting turn by turn, deciding
    the next step only after seeing the last result.
  - planning (this file): the model commits to the entire ordered plan
    *before* execution starts; the loop that runs it is fixed and
    deterministic.

Flow for one goal:

    goal ── build_planner_chain() ──> ["step 1", "step 2", ...]   (planning)
         ── for each step: build_executor_chain() ──> step result  (execution)
              (each call also sees the goal, the full plan, and the
               results of every earlier step)
         ── build_synthesis_chain() ──> final answer               (synthesis)
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# Upper bound on how many steps a plan may contain. The planner is asked to
# stay under this, and _parse_plan() also truncates to it, so a runaway plan
# can't turn into an unbounded number of executor calls.
MAX_STEPS: int = 6

# Sample goals for testing. Each is deliberately multi-part so the planner
# has something real to decompose rather than a question it could answer in
# a single sentence.
SAMPLE_GOALS: list[str] = [
    "Plan a one-day self-guided walking tour of Rome for a first-time visitor "
    "who is interested in history and wants to keep walking distances short.",
    "Outline a small Python command-line app that converts a folder of CSV "
    "files into a single Excel workbook, one sheet per file.",
]


def _parse_plan(raw: str) -> list[str]:
    """Turn the planner LLM's text output into a clean list of step strings.

    The planner is asked for a numbered list, one step per line, but models
    are inconsistent about the exact prefix ("1.", "1)", "- ", "Step 1:"),
    about blank lines, and about trailing commentary. This normalises all of
    that: keep non-empty lines, strip a leading bullet/number/punctuation
    run, drop anything left empty, and cap the result at MAX_STEPS.
    """
    steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip a leading list marker: digits, ".", ")", "-", "*", and the
        # word "Step", plus any surrounding whitespace. lstrip() with a
        # character set removes every leading character that is in the set,
        # which covers "1.", "1)", "- ", "12. " etc. in one pass.
        cleaned = line.lstrip("0123456789.)-*  ")
        if cleaned.lower().startswith("step "):
            # Handle a "Step 3: do the thing" style prefix that survived the
            # lstrip above (the "S" stopped it).
            cleaned = cleaned[len("step "):].lstrip("0123456789:.)-  ")
        if cleaned:
            steps.append(cleaned)
    # Truncate rather than raise: a slightly-too-long plan is still usable,
    # we just don't want to execute all of it.
    if len(steps) > MAX_STEPS:
        logger.warning("Plan had %d steps; truncating to MAX_STEPS (%d)", len(steps), MAX_STEPS)
    return steps[:MAX_STEPS]


def build_planner_chain() -> Runnable[dict[str, str], list[str]]:
    """Build the chain that decomposes a goal into an ordered list of steps.

    This is the "planning" half of the pattern. The chain is
    `prompt | llm | StrOutputParser() | _parse_plan`, so `.invoke({"goal": ...})`
    returns a ready-to-iterate `list[str]`.
    """
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a planner. Break the user's goal into a short ordered "
                f"list of at most {MAX_STEPS} concrete steps that, carried out in "
                "order, achieve the goal. Each step must be self-contained and "
                "actionable on its own. Output only the list, one step per line, "
                "numbered 1., 2., 3., ... with no preamble, no sub-bullets, and "
                "no commentary after the list.",
            ),
            ("human", "Goal: {goal}"),
        ]
    )
    return planner_prompt | llm | StrOutputParser() | _parse_plan


def build_executor_chain() -> Runnable[dict[str, str], str]:
    """Build the chain that carries out a single step of the plan.

    Every call sees the original goal, the full plan (for context on where
    this step sits), the results of the steps already completed, and the one
    step it is being asked to do now. It returns just that step's result.
    """
    executor_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are executing one step of a plan. Use the goal, the full "
                "plan, and the results of the completed steps as context. Do "
                "only the current step — do not do later steps or repeat earlier "
                "ones. Respond with just that step's result, concisely.",
            ),
            (
                "human",
                "Goal:\n{goal}\n\n"
                "Full plan:\n{plan}\n\n"
                "Results so far:\n{completed}\n\n"
                "Current step:\n{step}",
            ),
        ]
    )
    return executor_prompt | llm | StrOutputParser()


def build_synthesis_chain() -> Runnable[dict[str, str], str]:
    """Build the chain that combines all step results into the final answer."""
    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are given a goal and the results of the steps taken to "
                "reach it. Combine them into a single coherent answer to the "
                "goal. Output only the answer, with no preamble.",
            ),
            ("human", "Goal:\n{goal}\n\nStep results:\n{completed}"),
        ]
    )
    return synthesis_prompt | llm | StrOutputParser()


def _format_completed(results: list[tuple[str, str]]) -> str:
    """Render the (step, result) pairs done so far as a text block for prompts."""
    if not results:
        return "(none yet)"
    return "\n\n".join(
        f"Step {i}: {step}\nResult: {result}"
        for i, (step, result) in enumerate(results, start=1)
    )


def run_planner(goal: str) -> str:
    """Run the plan -> execute-each-step -> synthesise loop for a single goal.

    The chains are stateless and rebuilt per call; the loop's only state is
    `completed`, the growing list of (step, result) pairs that every later
    executor call — and the final synthesis call — is given as context.

    Returns:
        The synthesised final answer, or the plain plan text if the planner
        produced no usable steps.
    """
    planner_chain = build_planner_chain()
    executor_chain = build_executor_chain()
    synthesis_chain = build_synthesis_chain()

    plan = planner_chain.invoke({"goal": goal})
    if not plan:
        logger.warning("Planner produced no steps for goal: %s", goal)
        return "Could not produce a plan for this goal."

    print(f"Plan ({len(plan)} steps):")
    for i, step in enumerate(plan, start=1):
        print(f"  {i}. {step}")
    print()
    logger.info("Planned %d steps for goal: %s", len(plan), goal)

    # `plan_text` is the same for every executor call, so format it once.
    plan_text = "\n".join(f"{i}. {step}" for i, step in enumerate(plan, start=1))

    completed: list[tuple[str, str]] = []
    for i, step in enumerate(plan, start=1):
        result = executor_chain.invoke(
            {
                "goal": goal,
                "plan": plan_text,
                "completed": _format_completed(completed),
                "step": step,
            }
        ).strip()
        completed.append((step, result))
        print(f"Step {i} result: {result}\n")
        logger.info("Executed step %d/%d", i, len(plan))

    final_answer = synthesis_chain.invoke(
        {"goal": goal, "completed": _format_completed(completed)}
    ).strip()
    logger.info("Synthesised final answer for goal: %s", goal)
    return final_answer


def handle_requests() -> None:
    """Main entry point: run the planning loop for all sample goals.

    Registered in the CLI (see src/examples/__main__.py `_run_planning` and
    src/app/cli.py `CMD_PLANNING`), invoked by:
        uv run python -m src.examples planning
    """
    logger.info("Running the planning example over %d sample goals", len(SAMPLE_GOALS))
    print("Running plan-and-execute agent over sample goals")

    # Each goal is an independent run — run_planner() starts from an empty
    # plan and empty results every time.
    for goal in SAMPLE_GOALS:
        print("=" * 60)
        print(f"Goal: {goal}\n")
        final_answer = run_planner(goal)
        print("-" * 60)
        print(f"Final answer:\n{final_answer}")
        print("=" * 60)
        print()
