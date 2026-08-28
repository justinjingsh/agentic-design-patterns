"""State Management example using AWS Bedrock and LangChain.

Demonstrates the **State Management** agentic pattern: instead of treating a
conversation as one ever-growing blob of text that gets re-sent to the model
every turn, the agent keeps an explicit, structured ``ConversationState``
object and threads that *same object* through every step. Each turn reads from
the state to build the model's context and then writes back to it, and the
state has an intentional lifecycle for each kind of information it holds.

Three tiers of memory live in the state, each with its own retention policy:

  - ``recent``  — short-term / working memory. The last few turns, kept
    **verbatim** so the model has exact wording for follow-ups ("what did I
    just say?"). Bounded: once it grows past ``MAX_RECENT_TURNS`` the oldest
    turn is evicted.
  - ``summary`` — compressed history. When a turn is evicted from ``recent``
    it isn't thrown away; a summarizer chain folds it into a running prose
    summary. This keeps the *gist* of an arbitrarily long conversation at a
    roughly fixed token cost.
  - ``facts``   — long-term memory. Durable, structured key/value facts about
    the user or task ("name: Dana", "destination: Japan") that an extractor
    chain pulls out of each exchange and promotes into a dict. These survive
    for the whole session regardless of how old the turn that produced them
    is, and are always injected into context.

Per-turn flow (all four steps mutate one ``ConversationState``):

    user_message
      1. RETRIEVE  build the prompt context from state:
                     facts (all)  +  summary (if any)  +  recent (verbatim)
      2. RESPOND   respond_chain(context, user_message) -> assistant_reply
      3. RECORD    append ("user", msg) and ("assistant", reply) to `recent`
      4. UPDATE    a) fact_chain(msg, reply)  -> merge new facts into `facts`
                   b) while len(recent) > MAX_RECENT_TURNS:
                        pop oldest turn, summary_chain(summary, turn) -> summary

How this differs from the other patterns in this repo:
  - reflection / prompt_chaining: their only "state" is a single string
    passed from step to step. Here the state is a typed object with several
    fields, each updated by a different policy.
  - tools / planning / multiagent: the model decides *what happens next*.
    Here the sequence of steps is completely fixed; what's interesting is how
    information is *retained, compressed, and promoted* between steps.
  - Every other example starts each sample item from a clean slate. Here the
    whole point is that state carries across items in the loop — turn 5 can
    answer a question using a fact first mentioned in turn 1, long after that
    turn was compressed out of the verbatim window.
"""

import logging
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# How many (user, assistant) turns to keep word-for-word before the oldest
# one is evicted into the rolling summary. Small here so the compression path
# is exercised within a short sample conversation; in a real app this would be
# tuned against the model's context window and latency budget.
MAX_RECENT_TURNS: int = 3

# The fact extractor emits exactly this token when an exchange contained
# nothing worth storing long-term. Checked by exact match so a reply that
# merely mentions the word doesn't get misread as "no facts".
NO_FACTS_TOKEN: str = "NONE"

# A scripted conversation where later turns deliberately depend on information
# from earlier ones. By the time the last message is processed, the first two
# turns have been pushed out of `recent` and exist only in `summary` / `facts`
# — so a correct answer there proves the state management is doing its job.
SAMPLE_CONVERSATION: list[str] = [
    "Hi, my name is Dana and I'm planning a trip to Japan this April.",
    "I'll be travelling with my two kids, aged 6 and 9.",
    "We mostly want a mix of city sightseeing and some outdoor time.",
    "What kind of weather should we pack for?",
    "Suggest one activity the kids would enjoy.",
    "Quick recap: who is coming on this trip, and where are we going?",
]


@dataclass
class ConversationState:
    """The single mutable object threaded through every turn.

    An instance is created once per conversation in ``run_conversation`` and
    handed to every helper below, which read and mutate it in place. Nothing
    else persists between turns — if it matters later, it lives here.
    """

    # Long-term memory: durable key -> value facts, always injected into
    # context. Grows over the session; entries are overwritten if a later
    # turn revises a fact, never dropped by age.
    facts: dict[str, str] = field(default_factory=dict)

    # Compressed history: a single running prose summary of every turn that
    # has already been evicted from `recent`. Empty until the first eviction.
    summary: str = ""

    # Short-term memory: the most recent turns kept verbatim, as
    # (role, text) pairs where role is "user" or "assistant". Length is
    # capped at 2 * MAX_RECENT_TURNS entries (a user + assistant line each).
    recent: list[tuple[str, str]] = field(default_factory=list)


def build_respond_chain() -> Runnable[dict[str, str], str]:
    """Chain that writes the assistant's reply for one turn.

    Takes the fully-rendered context strings plus the new user message and
    returns just the reply text. It never sees the raw ``ConversationState``
    — the caller flattens the state into ``facts`` / ``summary`` / ``recent``
    strings first (step 1, RETRIEVE), so this chain stays a plain
    ``prompt | llm | parser`` with no knowledge of how memory is stored.
    """
    respond_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful travel assistant. Use the KNOWN FACTS and "
                "CONVERSATION SUMMARY to stay consistent with everything the user "
                "has told you earlier, even if it is not repeated in the recent "
                "messages. Reply in one or two short sentences.",
            ),
            (
                "human",
                "KNOWN FACTS:\n{facts}\n\n"
                "CONVERSATION SUMMARY:\n{summary}\n\n"
                "RECENT MESSAGES:\n{recent}\n\n"
                "New user message:\n{user}",
            ),
        ]
    )
    return respond_prompt | llm | StrOutputParser()


def build_fact_chain() -> Runnable[dict[str, str], str]:
    """Chain that extracts durable facts from one user/assistant exchange.

    Returns newline-separated ``key: value`` lines (lower-case, snake-ish
    keys) for anything worth remembering for the rest of the session, or
    exactly ``NO_FACTS_TOKEN`` if there was nothing. ``_merge_facts`` parses
    this back into the ``facts`` dict. Keeping extraction as its own model
    call — rather than asking the responder to also emit facts — means the
    long-term store has a single, auditable writer.
    """
    fact_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You maintain a long-term memory of stable facts about the user and "
                "their trip (names, ages, who is travelling, destination, dates, firm "
                "preferences). From the exchange below, output ONLY the NEW or CHANGED "
                "facts, one per line as 'key: value' with a short lower_snake_case key. "
                "Do not repeat facts that are unchanged. If there is nothing durable to "
                f"store, output exactly {NO_FACTS_TOKEN}.",
            ),
            ("human", "User said:\n{user}\n\nAssistant replied:\n{reply}"),
        ]
    )
    return fact_prompt | llm | StrOutputParser()


def build_summary_chain() -> Runnable[dict[str, str], str]:
    """Chain that folds one evicted turn into the rolling summary.

    Takes the current ``summary`` (may be empty) and the ``turn`` text being
    pushed out of the verbatim window, and returns the new summary. Called
    once per eviction in ``_compress_recent`` so the summary stays roughly
    constant in size no matter how long the conversation runs.
    """
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You keep a running summary of a conversation. Given the existing "
                "summary and one older message that is about to drop out of the "
                "verbatim history, return an updated summary that still captures every "
                "important point. Keep it to a few sentences; do not just append. Output "
                "only the summary prose itself, with no preamble like 'Updated summary:'.",
            ),
            (
                "human",
                "Existing summary:\n{summary}\n\nMessage leaving the window:\n{turn}",
            ),
        ]
    )
    return summary_prompt | llm | StrOutputParser()


def _format_facts(state: ConversationState) -> str:
    """Render ``state.facts`` as lines for the prompt (RETRIEVE, part 1)."""
    if not state.facts:
        return "(none yet)"
    return "\n".join(f"- {key}: {value}" for key, value in state.facts.items())


def _format_recent(state: ConversationState) -> str:
    """Render ``state.recent`` verbatim for the prompt (RETRIEVE, part 3)."""
    if not state.recent:
        return "(no earlier messages)"
    return "\n".join(f"{role}: {text}" for role, text in state.recent)


def _merge_facts(state: ConversationState, raw: str) -> None:
    """Parse the fact chain's output and update ``state.facts`` in place.

    Unrecognised lines are skipped rather than raised on — the extractor is
    an LLM and its formatting is only mostly reliable, so this stays lenient
    the same way the planning example's ``_parse_plan`` does.
    """
    cleaned = raw.strip()
    if not cleaned or cleaned == NO_FACTS_TOKEN:
        return

    for line in cleaned.splitlines():
        line = line.strip().lstrip("-* ").strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        # Strip any wrapping quotes the model added around the value.
        value = value.strip().strip('"').strip("'").strip()
        if key and value:
            # Overwrite on repeat: a later turn correcting an earlier fact
            # ("actually the kids are 7 and 9") should win.
            state.facts[key] = value
            logger.info("Long-term fact stored: %s = %s", key, value)


def _compress_recent(state: ConversationState, summary_chain: Runnable[dict[str, str], str]) -> None:
    """Evict oldest turns past the cap, folding each into ``state.summary``.

    This is the promotion step from short-term to compressed memory. It runs
    every turn but usually does nothing; only once ``recent`` exceeds
    ``2 * MAX_RECENT_TURNS`` entries does it pop from the front and call the
    summary chain. A ``while`` (not an ``if``) so it still converges if the
    cap is lowered mid-run.
    """
    max_entries = 2 * MAX_RECENT_TURNS
    while len(state.recent) > max_entries:
        role, text = state.recent.pop(0)
        state.summary = summary_chain.invoke(
            {"summary": state.summary or "(empty)", "turn": f"{role}: {text}"}
        ).strip()
        logger.info("Evicted oldest turn into summary; summary now %d chars", len(state.summary))


def run_conversation(messages: list[str]) -> ConversationState:
    """Run the whole scripted conversation against one ``ConversationState``.

    The chains are built once and reused for every turn — they're stateless,
    so all continuity comes from the ``state`` object, not from the chains.
    Returns the final state so callers can inspect what was retained.
    """
    respond_chain = build_respond_chain()
    fact_chain = build_fact_chain()
    summary_chain = build_summary_chain()

    state = ConversationState()

    for turn_number, user_message in enumerate(messages, start=1):
        print(f"[turn {turn_number}] user: {user_message}")

        # 1. RETRIEVE — flatten the current state into prompt context.
        context = {
            "facts": _format_facts(state),
            "summary": state.summary or "(nothing summarised yet)",
            "recent": _format_recent(state),
            "user": user_message,
        }

        # 2. RESPOND — generate the reply from that context.
        reply = respond_chain.invoke(context).strip()
        print(f"[turn {turn_number}] assistant: {reply}\n")

        # 3. RECORD — append both halves of the exchange to short-term memory.
        state.recent.append(("user", user_message))
        state.recent.append(("assistant", reply))

        # 4a. UPDATE long-term memory — extract and merge durable facts.
        raw_facts = fact_chain.invoke({"user": user_message, "reply": reply})
        _merge_facts(state, raw_facts)

        # 4b. UPDATE compressed memory — evict anything now past the window.
        _compress_recent(state, summary_chain)

    return state


def handle_requests() -> None:
    """Main entry point: run the sample conversation and show the final state."""
    logger.info("Running state management conversation over %d turns", len(SAMPLE_CONVERSATION))
    print("=" * 60)
    print("State Management: one ConversationState threaded through every turn")
    print("=" * 60)

    final_state = run_conversation(SAMPLE_CONVERSATION)

    print("-" * 60)
    print("Final state after the conversation:")
    print(f"\nLong-term facts ({len(final_state.facts)}):")
    print(_format_facts(final_state))
    print("\nRolling summary of evicted turns:")
    print(final_state.summary or "(none)")
    print(f"\nVerbatim recent window ({len(final_state.recent)} entries, "
          f"cap {2 * MAX_RECENT_TURNS}):")
    print(_format_recent(final_state))
    print("=" * 60)
