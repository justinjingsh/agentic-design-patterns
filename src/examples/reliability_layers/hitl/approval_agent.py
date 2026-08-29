"""Human-in-the-Loop (HITL) example using AWS Bedrock and LangChain.

Demonstrates the **Human-in-the-Loop** agentic pattern: the agent runs the
same observe -> decide -> act tool-calling loop as the ``tools`` example, but
before any *side-effecting* tool runs the loop pauses and hands the proposed
call to a human, who can:

  - **approve** it as-is,
  - **edit** the arguments and let the edited version run, or
  - **reject** it, in which case the tool never runs and the model is told so.

The agent can also *pull* a human in on its own initiative: a
``request_human_input`` tool lets it ask a clarifying question instead of
guessing when a request is under-specified.

Which tools need review is a fixed policy, not the model's choice:

  - ``search_flights`` / ``get_fare_rules`` — read-only, run automatically.
  - ``book_flight`` / ``send_email`` / ``cancel_booking`` — spend money, send
    external mail, or are irreversible, so every call goes through
    ``reviewer.decide(...)`` first (``SENSITIVE_TOOLS``).

The decision comes from a ``Reviewer``:

  - ``ScriptedReviewer`` — returns pre-canned decisions per tool name, so the
    CLI demo runs unattended and deterministically (same idea as the scripted
    flakiness in ``exception_handling``).
  - ``ConsoleReviewer`` — blocks on ``input()`` and asks a real person. Not
    used by ``handle_requests()``; swap it in
    (``run_agent(query, ConsoleReviewer())``) to drive the gate by hand.

Per tool call, inside ``_dispatch_call``:

    _dispatch_call(call, reviewer)
      ├─ name == "request_human_input"  ─> reviewer.decide(...) ─> "HUMAN_RESPONSE: <answer>"
      ├─ name not in SENSITIVE_TOOLS    ─> run it now ─────────> tool result
      └─ name in SENSITIVE_TOOLS        ─> reviewer.decide(...)
            APPROVE ─> run with original args ────────────────> tool result
            EDIT    ─> run with human's args ─────────────────> "HUMAN_EDITED: ... <result>"
            REJECT  ─> do not run ───────────────────────────> "HUMAN_REJECTED: <reason>"

How this differs from the neighbouring patterns in this repo:
  - tools: identical loop, but every tool call runs unconditionally. Here a
    policy-selected subset is intercepted and only proceeds with human sign-off.
  - exception_handling: also wraps each tool call, but to recover from a call
    that *fails*. Here the call would succeed — the concern is that it
    *should not happen* without a person in the loop.
  - reflection / goal_monitoring: an automated critic judges a finished draft.
    Here the judge is a human and the checkpoint is *before* the action, not
    after the output.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

# Message classes for the hand-rolled tool-calling loop — see the `tools`
# example for the full walkthrough of how these map to chat roles.
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

# Shared ChatBedrock instance, constructed once at import time in
# src/app/bedrock.py (which also validates AWS credentials at module scope).
from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# Caps model invocations for one request, exactly as in the `tools` example.
MAX_STEPS: int = 6

# Prefixes the driver puts on a tool result so the model can tell what the
# human did. They are only a convention the system prompt explains — nothing
# parses them except the model.
REJECTED_PREFIX: str = "HUMAN_REJECTED"
EDITED_PREFIX: str = "HUMAN_EDITED"
RESPONSE_PREFIX: str = "HUMAN_RESPONSE"

# The three shapes a human review can take.
APPROVE: str = "approve"
EDIT: str = "edit"
REJECT: str = "reject"
# Returned by the reviewer when answering a request_human_input question.
ANSWER: str = "answer"


@dataclass
class ReviewDecision:
    """One human decision about a proposed tool call (or a question answered).

    - ``APPROVE``: run the tool with the arguments the model proposed.
    - ``EDIT``: run the tool, but with ``edited_args`` instead.
    - ``REJECT``: do not run the tool; ``message`` is the reason, surfaced to
      the model so it can back off gracefully.
    - ``ANSWER``: not about a tool at all — ``message`` is the human's reply to
      a ``request_human_input`` question.
    """

    action: str
    edited_args: Optional[dict] = None
    message: str = ""


class Reviewer(Protocol):
    """Anything that can answer a human-review request.

    ``run_agent`` depends only on this method, so the scripted reviewer used by
    the demo and a real interactive one are interchangeable.
    """

    def decide(self, *, tool_name: str, args: dict, prompt: str) -> ReviewDecision:
        ...


def _print_review(
    tool_name: str, args: dict, decision: ReviewDecision
) -> None:
    """Echo what a human saw and chose, so the CLI output tells the story."""
    print(f"    | review requested: {tool_name}({args})")
    if decision.action == REJECT:
        print(f"    | -> REJECT: {decision.message}")
    elif decision.action == EDIT:
        print(f"    | -> EDIT args -> {decision.edited_args}")
    elif decision.action == ANSWER:
        print(f"    | -> ANSWER: {decision.message}")
    else:
        print("    | -> APPROVE")


@dataclass
class ScriptedReviewer:
    """Reviewer that replays canned decisions, keyed by tool name (FIFO).

    Lets the CLI demo run without a person present and produce the same output
    every time. If a tool is reviewed more often than the script expects, the
    safe default wins: deny a side-effecting call, and tell the agent to use
    its judgement on a question.
    """

    script: dict[str, list[ReviewDecision]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy so replaying (pop) doesn't mutate the caller's dict/lists.
        self._queues: dict[str, list[ReviewDecision]] = {
            name: list(decisions) for name, decisions in self.script.items()
        }

    def decide(self, *, tool_name: str, args: dict, prompt: str) -> ReviewDecision:
        queue = self._queues.get(tool_name)
        if queue:
            decision = queue.pop(0)
        elif tool_name == ASK_HUMAN_TOOL:
            decision = ReviewDecision(
                ANSWER,
                message="(no scripted answer — proceed with your best judgement "
                "and state your assumptions)",
            )
        else:
            decision = ReviewDecision(
                REJECT, message="No human available to approve; denied by default."
            )
        _print_review(tool_name, args, decision)
        return decision


class ConsoleReviewer:
    """Reviewer that blocks on stdin and asks a real person.

    Not wired into ``handle_requests()`` (the CLI demo must run unattended),
    but drop it in with ``run_agent(query, ConsoleReviewer())`` to approve,
    edit, or reject each call yourself.
    """

    def decide(self, *, tool_name: str, args: dict, prompt: str) -> ReviewDecision:
        print("\n>>> HUMAN REVIEW NEEDED")
        if tool_name == ASK_HUMAN_TOOL:
            answer = input(f"    Agent asks: {prompt}\n    Your answer: ").strip()
            return ReviewDecision(ANSWER, message=answer or "(no answer given)")

        print(f"    Tool: {tool_name}")
        print(f"    Args: {args}")
        choice = input("    [a]pprove / [e]dit / [r]eject? ").strip().lower()
        if choice.startswith("a"):
            return ReviewDecision(APPROVE)
        if choice.startswith("e"):
            raw = input(
                f"    New args as key=value pairs (comma-separated), "
                f"blank keeps {args}: "
            ).strip()
            return ReviewDecision(EDIT, edited_args=_parse_kv(raw, args))
        reason = input("    Reason for rejecting: ").strip()
        return ReviewDecision(REJECT, message=reason or "Rejected by the human.")


def _parse_kv(raw: str, base: dict) -> dict:
    """Parse ``a=1, b=two`` into a dict merged over ``base`` (ints coerced)."""
    merged = dict(base)
    for pair in raw.split(","):
        key, sep, value = pair.partition("=")
        if not sep:
            continue
        value = value.strip()
        merged[key.strip()] = int(value) if value.isdigit() else value
    return merged


# --- Offline data sources ----------------------------------------------------
# In-process so the example is deterministic and needs no network.

_FLIGHTS: list[dict] = [
    {"id": "AA100", "airline": "American", "depart": "07:15", "arrive": "15:45", "fare_usd": 291.0},
    {"id": "DL220", "airline": "Delta", "depart": "09:40", "arrive": "18:05", "fare_usd": 264.0},
    {"id": "B6915", "airline": "JetBlue", "depart": "13:05", "arrive": "21:20", "fare_usd": 248.0},
    {"id": "UA870", "airline": "United", "depart": "18:30", "arrive": "02:55", "fare_usd": 233.0},
]

# Seeded with one existing booking so the "cancel" scenario has something real
# to target. book_flight adds to this dict at runtime.
_BOOKINGS: dict[str, dict] = {
    "BK-2101": {"flight": "AA100", "passenger": "Sam Rivera", "status": "confirmed"},
}

_confirmation_seq: int = 4310


def _next_confirmation() -> str:
    """Deterministic booking codes: BK-4311, BK-4312, ..."""
    global _confirmation_seq
    _confirmation_seq += 1
    return f"BK-{_confirmation_seq}"


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """List available flights for a route and date. Read-only."""
    header = f"{origin.strip().upper()} -> {destination.strip().upper()} on {date}:"
    lines = [
        f"  {f['id']} {f['airline']:<9} dep {f['depart']} arr {f['arrive']} "
        f"{f['fare_usd']:.2f} USD"
        for f in _FLIGHTS
    ]
    return "\n".join([header, *lines])


@tool
def get_fare_rules(flight_id: str) -> str:
    """Return the change / cancellation policy for a flight. Read-only."""
    fid = flight_id.strip().upper()
    if not any(f["id"] == fid for f in _FLIGHTS):
        return f"no fare rules on file for {fid!r}"
    return (
        f"{fid}: changes 75 USD + fare difference; refundable up to 24h before "
        "departure, non-refundable thereafter."
    )


@tool
def book_flight(flight_id: str, passenger_name: str) -> str:
    """Reserve a seat and charge the fare. Spends money — needs human approval."""
    fid = flight_id.strip().upper()
    flight = next((f for f in _FLIGHTS if f["id"] == fid), None)
    if flight is None:
        return f"ERROR: no flight {fid!r} to book"
    code = _next_confirmation()
    _BOOKINGS[code] = {
        "flight": fid,
        "passenger": passenger_name,
        "status": "confirmed",
    }
    return (
        f"booked {fid} for {passenger_name} — confirmation {code}, "
        f"fare {flight['fare_usd']:.2f} USD"
    )


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email on the user's behalf. External side effect — needs approval."""
    return f"email sent to {to} (subject {subject!r}, {len(body)} chars)"


@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking. Irreversible — needs human approval."""
    bid = booking_id.strip().upper()
    booking = _BOOKINGS.get(bid)
    if booking is None:
        return f"ERROR: no booking {bid!r}"
    booking["status"] = "cancelled"
    return f"booking {bid} cancelled"


@tool
def request_human_input(question: str) -> str:
    """Ask the user a question when the request is ambiguous or missing a detail
    you need (origin, date, passenger, recipient, ...). Prefer this over
    guessing. The loop routes the question to a human and feeds their reply
    back to you as the tool result."""
    return "(routed to a human)"


# Every tool the model can see. request_human_input is a real bound tool so the
# model can choose to call it; the loop special-cases its "execution".
MODEL_TOOLS = [
    search_flights,
    get_fare_rules,
    book_flight,
    send_email,
    cancel_booking,
    request_human_input,
]
TOOLS_BY_NAME = {t.name: t for t in MODEL_TOOLS}

# Fixed policy: these tools never run without a human decision first.
SENSITIVE_TOOLS: set[str] = {
    book_flight.name,
    send_email.name,
    cancel_booking.name,
}
ASK_HUMAN_TOOL: str = request_human_input.name


SYSTEM_PROMPT: str = (
    "You are a travel assistant. You may call search_flights and get_fare_rules "
    "freely, but booking a flight, cancelling a booking, and sending email are "
    "actions a human must approve before they take effect.\n"
    "- Propose those actions by calling the matching tool with complete "
    "arguments; a human reviews the call before it runs.\n"
    f"- A tool result starting with '{REJECTED_PREFIX}:' means the human blocked "
    "that action. Do not retry it or route around it — tell the user it was not "
    "done and relay the reason.\n"
    f"- A result starting with '{EDITED_PREFIX}:' means the human approved the "
    "action but changed the arguments; report what actually happened using the "
    "adjusted values.\n"
    f"- A result starting with '{RESPONSE_PREFIX}:' is the human's answer to a "
    "question you asked with request_human_input.\n"
    "- If the request is missing something you need (origin, date, passenger "
    "name, email recipient) do not guess — call request_human_input with one "
    "specific question. When you have enough to answer, reply without calling "
    "a tool."
)


# Each sample request is paired with the script its reviewer replays, and is
# shaped to drive exactly one branch of the review logic.
SAMPLE_SESSIONS: list[tuple[str, dict[str, list[ReviewDecision]]]] = [
    # 1. APPROVE — the human OKs the booking exactly as proposed.
    (
        "Book the cheapest flight from SFO to JFK on 2026-09-14 for passenger "
        "Jordan Lee, then tell me the confirmation code.",
        {"book_flight": [ReviewDecision(APPROVE)]},
    ),
    # 2. EDIT — the human keeps the action but rewrites the arguments (fixes a
    #    mistyped recipient, tightens the subject) before it runs.
    (
        "Email the itinerary to my assistant at 'assist@example.com' with the "
        "subject 'itinerary'.",
        {
            "send_email": [
                ReviewDecision(
                    EDIT,
                    edited_args={
                        "to": "assistant@example.com",
                        "subject": "Itinerary: Jordan Lee, SFO-JFK, 2026-09-14",
                        "body": "Hi — forwarding the confirmed itinerary below.",
                    },
                )
            ]
        },
    ),
    # 3. REJECT — the human blocks an irreversible action; the agent must back
    #    off and report it, not look for another way to get it done.
    (
        "Cancel booking BK-2101 for me.",
        {
            "cancel_booking": [
                ReviewDecision(
                    REJECT,
                    message="Customer is still deciding; do not cancel yet.",
                )
            ]
        },
    ),
    # 4. CLARIFY — the request is under-specified, so the agent is expected to
    #    call request_human_input rather than guess; the human's answer unblocks
    #    it (and tells it not to book).
    (
        "Book me a flight to New York sometime next week.",
        {
            ASK_HUMAN_TOOL: [
                ReviewDecision(
                    ANSWER,
                    message=(
                        "Depart SFO on 2026-09-14, morning preferred. Just show "
                        "me the options for now — do not book anything yet."
                    ),
                )
            ]
        },
    ),
]


def _run_tool(name: str, args: dict) -> str:
    """Invoke a tool by name, returning errors as strings so the loop lives."""
    tool_fn = TOOLS_BY_NAME.get(name)
    if tool_fn is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        return str(tool_fn.invoke(args))
    except Exception as exc:  # noqa: BLE001 - example keeps the loop alive
        logger.exception("Tool %s raised", name)
        return f"ERROR: {name} failed ({exc})"


def _dispatch_call(call: dict, reviewer: Reviewer) -> str:
    """Run one tool call, inserting a human checkpoint where policy requires it.

    Always returns a string for the ``ToolMessage`` — a rejection is a result,
    not an error.
    """
    name = call["name"]
    args = call["args"]

    # The model explicitly wants to ask the human something.
    if name == ASK_HUMAN_TOOL:
        question = str(args.get("question", "")).strip()
        decision = reviewer.decide(tool_name=name, args=args, prompt=question)
        return f"{RESPONSE_PREFIX}: {decision.message}"

    # Read-only tool: no checkpoint, run it now.
    if name not in SENSITIVE_TOOLS:
        return _run_tool(name, args)

    # Side-effecting tool: pause for human review before anything happens.
    decision = reviewer.decide(
        tool_name=name,
        args=args,
        prompt=f"about to run side-effecting tool {name!r}",
    )

    if decision.action == REJECT:
        reason = decision.message or "The human declined this action."
        return (
            f"{REJECTED_PREFIX}: {reason} Do not retry it; tell the user it was "
            "not done and why."
        )
    if decision.action == EDIT and decision.edited_args is not None:
        result = _run_tool(name, decision.edited_args)
        return (
            f"{EDITED_PREFIX}: a human adjusted the arguments to "
            f"{decision.edited_args}. {result}"
        )
    # APPROVE (and any unexpected action) — run with the original arguments.
    return _run_tool(name, args)


def run_agent(query: str, reviewer: Reviewer) -> str:
    """Run the human-in-the-loop tool-calling loop for one request.

    Structurally identical to the ``tools`` example's ``run_agent`` — a bounded
    observe -> decide -> act loop over a growing message list — except each tool
    call goes through ``_dispatch_call``, which may pause for ``reviewer``.
    """
    model_with_tools = llm.bind_tools(MODEL_TOOLS)

    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(query),
    ]

    for step in range(1, MAX_STEPS + 1):
        ai_message: AIMessage = model_with_tools.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            logger.info("Agent finished after %d step(s)", step)
            return str(ai_message.content)

        for call in ai_message.tool_calls:
            print(f"  [tool] {call['name']}({call['args']})")
            output = _dispatch_call(call, reviewer)
            print(f"  [result] {output}")
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    logger.warning("Agent hit MAX_STEPS (%d) without a final answer", MAX_STEPS)
    return "Stopped: reached the maximum number of tool-calling steps."


def handle_requests() -> None:
    """Main entry point: run the HITL agent over the sample sessions.

    Registered in the CLI (see src/examples/__main__.py `_run_hitl` and
    src/app/cli.py `CMD_HITL`), invoked by:
        uv run python -m src.examples hitl
    """
    logger.info("Running human-in-the-loop agent over %d sessions", len(SAMPLE_SESSIONS))
    print("Running human-in-the-loop approval agent over sample requests")
    print(
        "(the human is scripted here so the demo runs unattended; swap in "
        "ConsoleReviewer to answer the prompts yourself)"
    )

    for query, script in SAMPLE_SESSIONS:
        # A fresh reviewer per session so each script starts unconsumed.
        reviewer = ScriptedReviewer(script)
        print("=" * 60)
        print(f"Request: {query}")
        answer = run_agent(query, reviewer)
        print("-" * 60)
        print(f"Answer: {answer}")
        print("=" * 60)
        print()
