"""Agent-to-Agent (A2A) example using AWS Bedrock and LangChain.

Demonstrates the **Agent-to-Agent** production pattern: instead of one process
holding every capability, agents run as independent *services*, each publishing
a machine-readable **Agent Card** (its name, description, and skills). A client
agent — here the **orchestrator** — discovers those cards at runtime, picks a
remote agent whose skills fit the user's request, and delegates the work by
sending it a **Message** over a transport. The remote agent runs the job as a
**Task** with an explicit lifecycle and hands back structured results
(messages + artifacts). The orchestrator never sees the remote agent's prompt,
model, or code — only its card and the Task it returns.

This is the "productionised" cousin of the ``multiagent`` example: there a
supervisor routes between personas that share one in-process transcript; here
each agent is behind a protocol boundary, discovered via its card, and invoked
with a task envelope that can come back ``completed``, ``failed``, or
``input-required`` (the remote agent pausing to ask for a missing detail).

Task lifecycle a remote agent walks through:

    submitted ──> working ──> completed        (result ready)
                          └─> input-required   (needs a detail; orchestrator
                          │                     crafts a follow-up and re-sends
                          │                     against the *same* task_id)
                          └─> failed           (handler raised)

Flow for one request:

    user request
      │
      ▼
    ┌─ orchestrator (client agent) ───────────────────────────────┐
    │  1. discover:  registry.discover() -> [AgentCard, ...]       │
    │  2. dispatch:  LLM picks one card by its skills (or NONE)    │
    │  3. craft:     LLM writes the A2A message for that agent     │
    └───────────────────────────┬─────────────────────────────────┘
                                │  Message(role="user", ...)
                                ▼  RemoteConnection.send()
    ┌─ remote agent (separate service, own card) ─────────────────┐
    │  Task: submitted -> working -> completed / input-required   │
    │  returns Task(history=[...], artifacts=[...])               │
    └───────────────────────────┬─────────────────────────────────┘
                                │  Task
                                ▼
    orchestrator: on input-required, craft a follow-up and re-send;
                  on completed, synthesise the final answer from the
                  task's artifacts.

Simplifications that keep the example offline and deterministic:
  - The registry + ``RemoteConnection`` stand in for HTTP discovery
    (``/.well-known/agent.json``) and an HTTP POST to the agent's endpoint.
    ``connection.send()`` is a direct function call, but the orchestrator only
    ever touches cards and the returned ``Task`` — swap in a real transport and
    nothing above it changes.
  - Each remote agent grounds its LLM step in a small hard-coded data table
    (FX rates, city forecasts) so runs are reproducible.
  - A ``Message`` carries plain text, not the protocol's typed multi-part
    payloads; an ``Artifact`` is a named string.

How this differs from the neighbouring patterns in this repo:
  - multiagent: teammates share one transcript and a supervisor picks the next
    speaker. Here agents are separate services with cards; the orchestrator
    discovers them and delegates over a transport, seeing only their Tasks.
  - routing: an LLM classifies the input once, then Python dispatches to a
    local handler. A2A also classifies (dispatch), but the target is a remote
    agent invoked with a task envelope that can ask for more input.
  - tools: the model calls in-process functions whose signatures it knows up
    front. Here capabilities are *discovered* from cards at runtime and run in
    another service with their own lifecycle.
  - hitl: wraps each call to gate a side effect. A2A wraps the call to cross a
    service boundary and carry a task through its states.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# Shared ChatBedrock instance, constructed once at import time in
# src/app/bedrock.py (which also validates AWS credentials at module scope).
from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# --- Protocol vocabulary ---------------------------------------------------
# Task lifecycle states. Plain string constants (the repo avoids Enum); the
# orchestrator branches on these after every `send`.
SUBMITTED: str = "submitted"
WORKING: str = "working"
INPUT_REQUIRED: str = "input-required"
COMPLETED: str = "completed"
FAILED: str = "failed"

# The dispatcher emits exactly this when no registered agent's skills cover the
# request. Checked by exact match so an agent name in a sentence can't be
# mistaken for a routing decision.
NO_AGENT_TOKEN: str = "NONE"

# Upper bound on orchestrator <-> remote-agent round trips for one request.
# Each `input-required` reply costs one extra round; this caps the loop.
MAX_ROUNDS: int = 3

_task_seq: int = 1000


def _new_task_id() -> str:
    """Deterministic, monotonic task ids: task-1001, task-1002, ..."""
    global _task_seq
    _task_seq += 1
    return f"task-{_task_seq}"


@dataclass
class AgentCard:
    """The public capability descriptor an agent publishes for discovery.

    Stands in for the JSON served at ``/.well-known/agent.json``. The
    orchestrator's dispatcher sees only this — never the agent's prompt or code.
    """

    name: str
    description: str
    skills: list[str]


@dataclass
class Message:
    """One turn on a task's conversation. ``role`` is ``"user"`` (the client)
    or ``"agent"`` (the remote service)."""

    role: str
    content: str


@dataclass
class Artifact:
    """A named output a remote agent attaches to a completed task."""

    name: str
    content: str


@dataclass
class Task:
    """The envelope a remote agent works and returns.

    ``history`` grows across round trips (the orchestrator re-sends against the
    same task on ``input-required``); ``artifacts`` are the structured results;
    ``error`` is set only when ``state == FAILED``.
    """

    task_id: str
    state: str
    history: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    error: str = ""


@dataclass
class HandlerResult:
    """What a remote agent's pure business logic returns to the service wrapper.

    Exactly one of ``answer`` (-> COMPLETED) or ``question`` (-> INPUT_REQUIRED)
    is set; raising from the handler maps to FAILED.
    """

    answer: str = ""
    question: str = ""
    artifacts: list[Artifact] = field(default_factory=list)


# Type of a remote agent's business logic: incoming message text -> result.
Handler = Callable[[str], HandlerResult]


# --- Remote-agent side (each of these would be its own service) -----------


def _phrase_chain(system_prompt: str) -> Runnable[dict, str]:
    """A stateless ``prompt | llm | parser`` chain used inside a handler to turn
    grounded data into a sentence or two. Rebuilt per call, like ``reflection``."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    return prompt | llm | StrOutputParser()


class RemoteAgentService:
    """Wraps an ``AgentCard`` + a ``Handler`` and drives the task lifecycle.

    ``execute`` is the only method the transport calls. It moves the task
    ``submitted -> working``, runs the handler, and lands on ``completed``,
    ``input-required``, or ``failed`` — always returning the Task, never raising.
    """

    def __init__(self, card: AgentCard, handler: Handler) -> None:
        self.card = card
        self.handler = handler

    def execute(self, incoming: Message, task: Optional[Task] = None) -> Task:
        if task is None:
            task = Task(task_id=_new_task_id(), state=SUBMITTED, history=[incoming])
        else:
            # Continuation of an input-required task: same id, appended turn.
            task.history.append(incoming)
        self._set_state(task, WORKING)

        try:
            result = self.handler(incoming.content)
        except Exception as exc:  # noqa: BLE001 - a failed task is a result here
            task.error = str(exc)
            logger.exception("[%s] handler raised", self.card.name)
            self._set_state(task, FAILED)
            return task

        if result.question:
            task.history.append(Message("agent", result.question))
            self._set_state(task, INPUT_REQUIRED)
            return task

        task.history.append(Message("agent", result.answer))
        task.artifacts.extend(result.artifacts)
        self._set_state(task, COMPLETED)
        return task

    def _set_state(self, task: Task, state: str) -> None:
        prev, task.state = task.state, state
        print(f"    [{self.card.name}] {task.task_id}: {prev} -> {state}")


# 1. FX Rate Agent -------------------------------------------------------------

_RATES_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.09,
    "GBP": 1.27,
    "JPY": 0.0067,
    "AUD": 0.66,
    "CAD": 0.74,
}

# Words the agent will accept in place of a 3-letter code.
_CURRENCY_WORDS: dict[str, str] = {
    "dollar": "USD", "dollars": "USD", "buck": "USD", "bucks": "USD",
    "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "yen": "JPY",
}

_FX_SYSTEM: str = (
    "You are the FX Rate Agent, a standalone service. You are given the user's "
    "conversion request and a pre-computed result line. Confirm the conversion "
    "in one or two plain sentences. Do not invent rates or add caveats beyond "
    "the numbers you are given."
)


def _find_amount(text: str) -> Optional[float]:
    """First number in the text (commas allowed), or None."""
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return float(match.group().replace(",", "")) if match else None


def _find_currencies(text: str) -> list[str]:
    """Currency codes mentioned in the text, in first-appearance order, deduped.

    Accepts both 3-letter codes (``EUR``) and plain words (``euros``, ``yen``).
    """
    hits: list[tuple[int, str]] = []
    for match in re.finditer(r"\b([A-Za-z]{3,8})\b", text):
        word = match.group(1)
        code = None
        if word.upper() in _RATES_TO_USD:
            code = word.upper()
        elif word.lower() in _CURRENCY_WORDS:
            code = _CURRENCY_WORDS[word.lower()]
        if code:
            hits.append((match.start(), code))

    ordered: list[str] = []
    for _, code in sorted(hits):
        if code not in ordered:
            ordered.append(code)
    return ordered


def _fx_handler(text: str) -> HandlerResult:
    """Convert an amount between two currencies, or ask for what's missing."""
    amount = _find_amount(text)
    currencies = _find_currencies(text)

    if amount is None:
        return HandlerResult(
            question="What amount would you like converted, and between which "
            "currencies?"
        )
    if len(currencies) < 2:
        if currencies:
            return HandlerResult(
                question=f"I can see the target currency ({currencies[0]}) but "
                f"not the source. Which currency is the {amount:g} in?"
            )
        return HandlerResult(
            question=f"Which two currencies should I convert {amount:g} between "
            "(e.g. 'EUR to USD')?"
        )

    source, target = currencies[0], currencies[1]
    rate = _RATES_TO_USD[source] / _RATES_TO_USD[target]
    converted = amount * rate
    line = (
        f"{amount:g} {source} = {converted:,.2f} {target} "
        f"(rate {rate:.4f} {target}/{source})"
    )
    prose = _phrase_chain(_FX_SYSTEM).invoke(
        {"input": f"User asked: {text}\nComputed: {line}"}
    ).strip()
    return HandlerResult(answer=prose, artifacts=[Artifact("conversion", line)])


_FX_CARD = AgentCard(
    name="fx-rate-agent",
    description="Converts an amount of money from one currency to another using "
    "daily reference rates.",
    skills=["currency conversion", "exchange-rate lookup", "multi-currency amounts"],
)


# 2. Weather Agent -----------------------------------------------------------

_FORECASTS: dict[str, list[str]] = {
    "kyoto": [
        "Fri: 24C, light rain in the afternoon",
        "Sat: 22C, overcast, breezy",
        "Sun: 27C, sunny",
    ],
    "reykjavik": [
        "Fri: 8C, gusty wind and showers",
        "Sat: 6C, sleet",
        "Sun: 9C, partly cloudy",
    ],
    "singapore": [
        "Fri: 31C, humid, thunderstorms at night",
        "Sat: 32C, hazy sun",
        "Sun: 30C, afternoon downpours",
    ],
}

_WEATHER_SYSTEM: str = (
    "You are the Weather Agent, a standalone service. You are given a city and "
    "its 3-day forecast lines. Summarise the outlook in two or three sentences "
    "and suggest what to pack. Use only the forecast provided."
)


def _weather_handler(text: str) -> HandlerResult:
    """Report a city's forecast and a packing suggestion, or ask which city."""
    lowered = text.lower()
    city = next((name for name in _FORECASTS if name in lowered), None)
    if city is None:
        return HandlerResult(
            question="Which city do you want the forecast for? I currently cover "
            + ", ".join(sorted(name.title() for name in _FORECASTS))
            + "."
        )

    block = "\n".join(_FORECASTS[city])
    prose = _phrase_chain(_WEATHER_SYSTEM).invoke(
        {"input": f"City: {city.title()}\nForecast:\n{block}\n\nUser asked: {text}"}
    ).strip()
    return HandlerResult(
        answer=prose,
        artifacts=[Artifact("forecast", f"{city.title()}\n{block}")],
    )


_WEATHER_CARD = AgentCard(
    name="weather-agent",
    description="Reports the short-term forecast for a city and suggests what "
    "to pack.",
    skills=["city weather forecast", "packing suggestions", "trip weather"],
)


# --- Registry + transport (stand-in for discovery + HTTP) -----------------


@dataclass
class RemoteConnection:
    """Client-side handle to one remote agent — the wire.

    In a real A2A setup ``send`` is an HTTP POST to the agent's endpoint. Here
    it's a direct call, but the orchestrator only ever sees the returned Task.
    """

    _service: RemoteAgentService

    def send(self, message: Message, task: Optional[Task] = None) -> Task:
        return self._service.execute(message, task)


class AgentRegistry:
    """Where agents are registered and where the orchestrator discovers them.

    Stands in for a discovery service / catalogue of ``/.well-known/agent.json``
    documents.
    """

    def __init__(self) -> None:
        self._services: dict[str, RemoteAgentService] = {}

    def register(self, service: RemoteAgentService) -> None:
        self._services[service.card.name] = service
        logger.info(
            "registered agent %r (skills: %s)",
            service.card.name,
            ", ".join(service.card.skills),
        )

    def discover(self) -> list[AgentCard]:
        """The cards a client can see — no handlers, no internals."""
        return [service.card for service in self._services.values()]

    def connect(self, name: str) -> RemoteConnection:
        return RemoteConnection(self._services[name])


def build_registry() -> AgentRegistry:
    """A registry populated with the two demo agents."""
    registry = AgentRegistry()
    registry.register(RemoteAgentService(_FX_CARD, _fx_handler))
    registry.register(RemoteAgentService(_WEATHER_CARD, _weather_handler))
    return registry


# --- Orchestrator side (the client agent) --------------------------------


def _render_cards(cards: list[AgentCard]) -> str:
    """Render discovered cards as the menu the dispatcher LLM chooses from."""
    return "\n".join(
        f"- {card.name}: {card.description} (skills: {', '.join(card.skills)})"
        for card in cards
    )


def _render_artifacts(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return "(none)"
    return "\n\n".join(f"[{a.name}]\n{a.content}" for a in artifacts)


def build_dispatch_chain() -> Runnable[dict, str]:
    """Pick the one remote agent whose skills fit the request, or ``NONE``."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the dispatcher inside a client agent. You are given a "
                "user request and a list of remote agents, each with a "
                "description and skills. Choose the ONE agent whose skills best "
                "cover the request. Reply with exactly that agent's name and "
                f'nothing else. If no agent fits, reply with exactly "{NO_AGENT_TOKEN}".',
            ),
            ("human", "User request:\n{request}\n\nRemote agents:\n{roster}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_request_chain() -> Runnable[dict, str]:
    """Craft the message the orchestrator sends to the chosen remote agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a client agent preparing a message to send to a remote "
                "specialist agent over the A2A protocol. Given the user's "
                "request and the target agent's card, write the message to send: "
                "a single self-contained instruction with every detail the agent "
                "needs and nothing it doesn't. Output only the message text.",
            ),
            ("human", "User request:\n{request}\n\nTarget agent:\n{card}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_followup_chain() -> Runnable[dict, str]:
    """Answer a remote agent's ``input-required`` question and restate the task."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "The remote agent paused and asked for more information before "
                "it can finish. Given the user's original request, the message "
                "you last sent, and the agent's question, write a short "
                "replacement message that answers the question and restates the "
                "full task. If a needed detail was never given by the user, pick "
                "the most reasonable default and state it explicitly. Output "
                "only the message text.",
            ),
            (
                "human",
                "User request:\n{request}\n\nLast message you sent:\n{last_message}"
                "\n\nAgent's question:\n{question}",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_synthesis_chain() -> Runnable[dict, str]:
    """Relay a completed remote task's result back to the user."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the client agent relaying a remote specialist's result "
                "to the user. Given the user's request and the specialist's "
                "answer plus any artifacts, reply to the user directly and "
                "concisely. Trust the specialist's answer; do not second-guess "
                "its numbers.",
            ),
            (
                "human",
                "User request:\n{request}\n\nSpecialist answer:\n{answer}\n\n"
                "Artifacts:\n{artifacts}",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def _first_token(raw: str) -> str:
    """First word of the dispatcher reply, stripped of quotes/punctuation."""
    cleaned = raw.strip().strip("\"'`.\n ")
    return cleaned.split()[0] if cleaned else NO_AGENT_TOKEN


def _decline(cards: list[AgentCard]) -> str:
    """Message when discovery turns up no agent that covers the request."""
    menu = "\n".join(f"  - {c.name}: {c.description}" for c in cards)
    return (
        "No registered agent has a skill that covers this request. "
        f"Registered agents:\n{menu}"
    )


def run_orchestrator(registry: AgentRegistry, user_request: str) -> str:
    """Discover -> dispatch -> delegate -> (follow up) -> synthesise, for one request.

    The orchestrator's only state is the current ``Task`` handle and the last
    ``Message`` it sent; everything else lives behind ``RemoteConnection``.
    """
    cards = registry.discover()
    print(f"  discover: {[c.name for c in cards]}")

    raw_choice = build_dispatch_chain().invoke(
        {"request": user_request, "roster": _render_cards(cards)}
    )
    choice = _first_token(raw_choice)
    print(f"  dispatch: {choice!r} (raw {raw_choice.strip()!r})")

    names = {c.name for c in cards}
    if choice not in names:
        print("  -> no capable agent; orchestrator declines")
        return _decline(cards)

    card = next(c for c in cards if c.name == choice)
    connection = registry.connect(choice)
    followup_chain = build_followup_chain()

    content = build_request_chain().invoke(
        {"request": user_request, "card": _render_cards([card])}
    ).strip()
    message = Message("user", content)
    print(f"  send -> {choice}: {content!r}")
    task = connection.send(message)

    rounds = 1
    while task.state == INPUT_REQUIRED and rounds < MAX_ROUNDS:
        question = task.history[-1].content
        print(f"  {choice} asks: {question!r}")
        content = followup_chain.invoke(
            {
                "request": user_request,
                "last_message": message.content,
                "question": question,
            }
        ).strip()
        message = Message("user", content)
        print(f"  send -> {choice} (round {rounds + 1}): {content!r}")
        task = connection.send(message, task)
        rounds += 1

    if task.state == COMPLETED:
        final = build_synthesis_chain().invoke(
            {
                "request": user_request,
                "answer": task.history[-1].content,
                "artifacts": _render_artifacts(task.artifacts),
            }
        ).strip()
        return final

    if task.state == INPUT_REQUIRED:
        return (
            f"Could not finish: {choice} still needs more information after "
            f"{rounds} round(s) — {task.history[-1].content}"
        )
    return (
        f"Could not finish: {choice} returned state {task.state!r}"
        + (f" ({task.error})" if task.error else "")
    )


# Each sample drives one path through the orchestrator.
SAMPLE_REQUESTS: list[str] = [
    # 1. Clean delegation -> completed.
    "How much is 250 EUR in USD right now?",
    # 2. A different agent, also completed.
    "I land in Kyoto on Friday for three days — what's the weather like and "
    "what should I pack?",
    # 3. Under-specified: the fx agent replies input-required (no source
    #    currency), the orchestrator supplies a default and re-sends.
    "Convert 300 into Japanese yen.",
    # 4. No registered agent has a matching skill -> orchestrator declines.
    "Recommend a good novel to read on the flight.",
]


def handle_requests() -> None:
    """Main entry point: run the A2A orchestrator over the sample requests.

    Registered in the CLI (see src/examples/__main__.py `_run_a2a` and
    src/app/cli.py `CMD_A2A`), invoked by:
        uv run python -m src.examples a2a
    """
    logger.info("Running the A2A example over %d sample requests", len(SAMPLE_REQUESTS))
    print("Running Agent-to-Agent orchestration over sample requests")

    registry = build_registry()
    print(f"Registry: {[c.name for c in registry.discover()]}\n")

    for request in SAMPLE_REQUESTS:
        print("=" * 60)
        print(f"Request: {request}")
        answer = run_orchestrator(registry, request)
        print("-" * 60)
        print(f"Answer: {answer}")
        print("=" * 60)
        print()
