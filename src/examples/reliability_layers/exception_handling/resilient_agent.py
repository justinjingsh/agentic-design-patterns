"""Exception Handling example using AWS Bedrock and LangChain.

Demonstrates the **Exception Handling** (a.k.a. error recovery / resilience)
agentic pattern: the agent's control flow is the same observe -> decide ->
act -> repeat tool-calling loop as the ``tools`` example, but every tool
execution is wrapped in a recovery ladder so that a failing tool degrades the
answer instead of crashing the run.

The recovery ladder applied to each tool call, in order:

  1. **Classify** the failure. Tools raise either ``TransientError`` (a
     timeout / rate-limit / flaky-network style failure that a retry might
     fix) or ``PermanentError`` (bad input, missing resource, auth — retrying
     is pointless). Anything else that escapes a tool (a bug, an
     unanticipated dependency failure) is caught by a catch-all and treated
     as permanent, so the loop can never be killed by a surprise exception.
  2. **Retry with backoff.** ``TransientError`` is retried up to
     ``MAX_RETRIES`` times with exponential backoff
     (``BACKOFF_BASE_SECONDS * 2**attempt``). ``PermanentError`` skips
     straight past this step.
  3. **Fall back.** If retries are exhausted (or the error was permanent)
     and ``FALLBACKS`` names an alternative tool for this one, call that
     instead — transparently, so the model doesn't have to know the primary
     source went down. The fallback gets its own retry budget.
  4. **Degrade gracefully.** If there is no fallback, or the fallback also
     fails, return a plain-text ``TOOL_UNAVAILABLE: ...`` note as the tool
     result. The model reads that like any other ``ToolMessage`` and is told
     to answer with whatever it still has and flag the gap. ``run_agent()``
     never raises on a tool failure.

Per tool call:

    call_tool_with_recovery(name, args)
      ├─ _attempt_with_retries(name, args)          # retry transient, N times
      │     success ─────────────────────────────>  return result
      │     ToolError
      ├─ FALLBACKS[name] exists?
      │     yes ─ _attempt_with_retries(fallback)   # retry the fallback too
      │             success ─────────────────────>  return "[fallback: ...] result"
      │             ToolError ───────────────────>  _degraded(...)
      │     no ──────────────────────────────────>  _degraded(...)   # "TOOL_UNAVAILABLE: ..."

How this differs from the neighbouring patterns in this repo:
  - tools: the same hand-rolled tool-calling loop, but it assumes every tool
    call succeeds — one raised exception aborts the whole query. This example
    is that loop hardened: the mechanics on show are the failure taxonomy and
    the retry -> fallback -> degrade ladder, not the tool calling itself.
  - reflection / goal_monitoring: recover from a *low-quality* result by
    iterating. Here the concern is a step that *fails outright* — no output
    to critique, just an exception to classify and route.
"""

import logging
import time

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

# Caps the number of model invocations for one query, exactly as in the
# `tools` example — a model that keeps asking for tools can't spin forever.
MAX_STEPS: int = 6

# How many times a single tool call is attempted when it keeps raising
# TransientError. The count includes the first try, so MAX_RETRIES = 3 means
# "try once, then retry twice".
MAX_RETRIES: int = 3

# Base for the exponential backoff between transient retries: the wait before
# attempt N is BACKOFF_BASE_SECONDS * 2 ** (N - 1) (0.5s, 1.0s, 2.0s, ...).
# Kept small so the example runs quickly; a real system would use larger
# values plus random jitter to avoid thundering-herd retries.
BACKOFF_BASE_SECONDS: float = 0.5

# Prefix the driver puts on a degraded (unrecoverable) tool result. It's just
# a convention the system prompt tells the model about — there's nothing
# magic about the string.
UNAVAILABLE_PREFIX: str = "TOOL_UNAVAILABLE"


# --- Failure taxonomy ------------------------------------------------------
# Tools raise one of these so the driver can decide *how* to react without
# inspecting error messages. Everything else that escapes a tool is caught by
# a catch-all in `_attempt_with_retries` and re-wrapped as PermanentError.


class ToolError(Exception):
    """Base class for a tool failure the driver knows how to handle."""


class TransientError(ToolError):
    """A failure that might succeed if retried (timeout, rate limit, flaky I/O)."""


class PermanentError(ToolError):
    """A failure a retry won't fix (bad input, missing resource, auth denied)."""


# --- Offline data sources ------------------------------------------------------
# All kept in-process so the example is deterministic. The "flakiness" below
# is scripted (a call counter), not random, so every run fails and recovers
# in exactly the same places.

# Primary price feed. Deliberately missing GLOBEX so a query for it forces the
# fallback path.
_PRICES: dict[str, float] = {"ADP": 187.42, "ACME": 63.10}

# Backup price feed used only by the driver as a fallback for
# `fetch_stock_price`. Less fresh (results are tagged "delayed") but has wider
# coverage, including GLOBEX.
_BACKUP_PRICES: dict[str, float] = {"ADP": 185.55, "ACME": 62.80, "GLOBEX": 416.84}

# The primary feed "times out" this many times per ticker before it starts
# succeeding — this is what the retry step recovers from.
_TRANSIENT_FAILURES: int = 2
_price_calls: dict[str, int] = {}

_RATES: dict[str, float] = {"EUR": 0.92, "GBP": 0.79, "JPY": 156.0}


@tool
def fetch_stock_price(ticker: str) -> str:
    """Return the latest USD share price for a stock ticker such as "ADP"."""
    key = ticker.strip().upper()
    if key not in _PRICES:
        # Permanent for this feed: it simply doesn't carry this ticker. The
        # driver skips retries and routes to the backup feed instead.
        raise PermanentError(f"ticker {key!r} not covered by the primary feed")
    # Scripted transient failure: the first `_TRANSIENT_FAILURES` calls for
    # each ticker raise as if the upstream feed timed out; the next call
    # succeeds. Deterministic stand-in for a genuinely flaky dependency.
    _price_calls[key] = _price_calls.get(key, 0) + 1
    if _price_calls[key] <= _TRANSIENT_FAILURES:
        raise TransientError(
            f"price feed timed out (call {_price_calls[key]} for {key})"
        )
    return f"{key} = {_PRICES[key]:.2f} USD"


@tool
def fetch_stock_price_backup(ticker: str) -> str:
    """Fallback price source: quotes are delayed but availability is high."""
    key = ticker.strip().upper()
    if key not in _BACKUP_PRICES:
        raise PermanentError(f"backup feed has no data for {key!r}")
    return f"{key} = {_BACKUP_PRICES[key]:.2f} USD (delayed)"


@tool
def convert_currency(amount_usd: float, to_currency: str) -> str:
    """Convert a USD amount into another currency (supported: EUR, GBP, JPY)."""
    code = to_currency.strip().upper()
    if code not in _RATES:
        # Bad argument: no amount of retrying makes an unsupported currency
        # supported, so this is permanent and the retry step is skipped.
        raise PermanentError(f"unsupported currency {code!r}")
    return f"{amount_usd:.2f} USD = {amount_usd * _RATES[code]:.2f} {code}"


@tool
def get_market_news(ticker: str) -> str:
    """Return a one-line market-news headline for a ticker."""
    # Simulates a dependency that fails in a way this code did NOT anticipate:
    # a plain RuntimeError, not a ToolError. The catch-all in
    # `_attempt_with_retries` must absorb it (re-wrapped as PermanentError) so
    # the run continues; there is no fallback for this tool, so the call ends
    # in graceful degradation.
    raise RuntimeError("news provider returned an empty body with HTTP 200")


# Only the primary tools are advertised to the model. `fetch_stock_price_backup`
# is intentionally absent — the driver substitutes it behind the scenes so the
# model never has to reason about which data source is up.
PRIMARY_TOOLS = [fetch_stock_price, convert_currency, get_market_news]
TOOLS_BY_NAME = {
    t.name: t for t in [*PRIMARY_TOOLS, fetch_stock_price_backup]
}

# Primary tool name -> the tool to try when it is unrecoverable. A tool with
# no entry here goes straight from "failed" to "degraded".
FALLBACKS: dict[str, str] = {"fetch_stock_price": "fetch_stock_price_backup"}

# Sample queries, each shaped to drive one branch of the recovery ladder.
SAMPLE_QUERIES: list[str] = [
    # Retry: the primary price feed times out twice, then succeeds on the
    # third attempt — all inside a single tool call.
    "What is the current share price of ADP?",
    # Fallback: the primary feed doesn't carry GLOBEX (permanent), so the
    # driver transparently switches to the backup feed, then the model
    # converts the delayed quote to EUR.
    "How much is one GLOBEX share in EUR?",
    # Graceful degradation: the news tool throws an unexpected error and has
    # no fallback, so the agent reports the price and flags the missing news.
    "Give me ACME's price and its latest market news.",
]


def _run_tool_once(name: str, args: dict) -> str:
    """Invoke one tool by name. Raises PermanentError for an unknown name."""
    tool_fn = TOOLS_BY_NAME.get(name)
    if tool_fn is None:
        raise PermanentError(f"unknown tool {name!r}")
    return str(tool_fn.invoke(args))


def _attempt_with_retries(name: str, args: dict) -> str:
    """Call a tool, retrying only ``TransientError`` with exponential backoff.

    Returns the tool's string result on success. Raises ``ToolError`` if the
    call ultimately fails: a ``PermanentError`` is re-raised immediately, a
    ``TransientError`` is re-raised once the retry budget is spent, and any
    other exception is caught and re-raised as ``PermanentError`` so callers
    only ever have to handle ``ToolError``.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_tool_once(name, args)
        except PermanentError as exc:
            logger.warning("Tool %s: permanent error, not retrying: %s", name, exc)
            print(f"    ! {name}: permanent error ({exc}) -> no retry")
            raise
        except TransientError as exc:
            if attempt == MAX_RETRIES:
                logger.warning(
                    "Tool %s: still failing after %d attempts: %s", name, attempt, exc
                )
                print(f"    ! {name}: transient error, retries exhausted ({exc})")
                raise
            wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.info(
                "Tool %s: transient error on attempt %d/%d (%s); retrying in %.2fs",
                name, attempt, MAX_RETRIES, exc, wait,
            )
            print(
                f"    ! {name}: transient error (attempt {attempt}/{MAX_RETRIES}: {exc})"
                f" -> retry in {wait:.2f}s"
            )
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: unknown failures
            # A tool raised something outside the taxonomy. Don't retry a
            # failure we don't understand; log it with a stack trace and
            # re-wrap so the caller's `except ToolError` still catches it.
            logger.exception("Tool %s: unexpected error, treating as permanent", name)
            print(f"    ! {name}: unexpected {type(exc).__name__} ({exc}) -> permanent")
            raise PermanentError(f"unexpected error in {name}: {exc}") from exc

    # Loop can only exit via return or raise above; this satisfies type checkers.
    raise PermanentError(f"{name}: retry loop exited without a result")


def _degraded(name: str, exc: BaseException) -> str:
    """Last resort: a readable failure note the model can work around."""
    logger.error("Tool %s unrecoverable, degrading: %s", name, exc)
    print(f"    x {name}: unrecoverable -> degraded result")
    return (
        f"{UNAVAILABLE_PREFIX}: {name} could not be completed ({exc}). "
        "Answer using whatever other information you have and clearly note "
        "that this data is missing."
    )


def call_tool_with_recovery(name: str, args: dict) -> str:
    """Run a tool through the full retry -> fallback -> degrade ladder.

    Always returns a string and never raises, so a broken tool produces a
    usable ``ToolMessage`` instead of aborting ``run_agent()``.
    """
    try:
        result = _attempt_with_retries(name, args)
        print(f"    = {name}: ok")
        return result
    except ToolError as primary_exc:
        fallback_name = FALLBACKS.get(name)
        if not fallback_name:
            return _degraded(name, primary_exc)
        logger.info("Falling back from %s to %s", name, fallback_name)
        print(f"    ~ {name} failed ({primary_exc}) -> fallback to {fallback_name}")
        try:
            result = _attempt_with_retries(fallback_name, args)
            print(f"    = {fallback_name}: ok (fallback)")
            return f"[fallback: {fallback_name}] {result}"
        except ToolError as fallback_exc:
            return _degraded(fallback_name, fallback_exc)


def run_agent(query: str) -> str:
    """Run the resilient tool-calling loop for one query and return the answer.

    Structurally identical to the ``tools`` example's ``run_agent`` — a
    bounded observe -> decide -> act loop over a growing message list — except
    tool execution goes through ``call_tool_with_recovery`` instead of calling
    the tool directly.
    """
    model_with_tools = llm.bind_tools(PRIMARY_TOOLS)

    messages: list[BaseMessage] = [
        SystemMessage(
            "You are a market-data assistant. Use the provided tools to answer, "
            "and you may call more than one. A tool result beginning with "
            f"'{UNAVAILABLE_PREFIX}:' means that data could not be fetched — do "
            "not retry it; answer with what you have and state plainly what is "
            "missing. A result tagged '[fallback: ...]' or '(delayed)' is from a "
            "backup source; use it but mention it may be slightly stale. When you "
            "have enough information, reply directly without calling a tool."
        ),
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
            # The whole point of this example: this call cannot raise, no
            # matter how the tool fails.
            output = call_tool_with_recovery(call["name"], call["args"])
            print(f"  [result] {output}")
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    logger.warning("Agent hit MAX_STEPS (%d) without a final answer", MAX_STEPS)
    return "Stopped: reached the maximum number of tool-calling steps."


def handle_requests() -> None:
    """Main entry point: run the resilient agent over the sample queries.

    Registered in the CLI (see src/examples/__main__.py `_run_exception_handling`
    and src/app/cli.py `CMD_EXCEPTION_HANDLING`), invoked by:
        uv run python -m src.examples exception_handling
    """
    logger.info("Running exception-handling agent over %d queries", len(SAMPLE_QUERIES))
    print("Running resilient tool-use agent over sample queries")

    # Each query is an independent run, but the module-level `_price_calls`
    # counter persists across them on purpose: once the primary feed has
    # "warmed up" for a ticker it keeps succeeding, just like a real one would.
    for query in SAMPLE_QUERIES:
        print("=" * 60)
        print(f"Query: {query}")
        answer = run_agent(query)
        print("-" * 60)
        print(f"Answer: {answer}")
        print("=" * 60)
        print()
