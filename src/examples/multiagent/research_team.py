"""Multi-agent collaboration example using AWS Bedrock and LangChain.

Demonstrates the Multi-Agent pattern: rather than one model answering a task,
a small team of single-purpose agents — each with its own persona and system
prompt — work the task on a shared transcript, and a *supervisor* agent
decides after every turn which teammate should act next (or that the team is
finished). A final deterministic pass then compiles the transcript into the
deliverable.

The supervisor is itself an agent: on each turn it reads the goal and
everything the team has said so far and picks the next speaker. That is what
makes this "collaboration" and not just a fan-out — each agent builds on the
others' output, and the routing between them is decided at runtime from the
evolving shared state.

How this differs from the other patterns in this repo:
  - prompt_chaining: a fixed two-step `|` chain; one model, hard-coded steps.
  - routing: the model classifies the input once, then deterministic Python
    dispatches to a single handler — there is no back-and-forth.
  - parallelization: independent sub-tasks fanned out and merged; the workers
    never see each other's output.
  - reflection: one draft improved by a generate -> reflect -> refine loop;
    the roles are fixed and there is no team.
  - planning: one model commits to an ordered plan up front, then a fixed
    loop executes it.
  - tools: one model interleaves thinking and tool calls turn by turn.
  - multiagent: several *distinct* agents share one transcript,
    and a supervisor agent chooses which of them speaks on each turn until
    the work is done.

Flow for one goal:

    goal ── supervisor ──> "researcher"   ── researcher agent ──> note
         ── supervisor ──> "analyst"      ── analyst agent    ──> note
         ── supervisor ──> "writer"       ── writer agent     ──> draft
         ── supervisor ──> "DONE"
         ── editor (deterministic final pass over the whole transcript)
              ──> final deliverable
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ...app.bedrock import llm

logger = logging.getLogger(__name__)

# Upper bound on supervisor turns. Each turn is one supervisor decision plus
# (usually) one specialist reply, so this caps how many model calls a single
# run can make even if the supervisor never says it is finished.
MAX_TURNS: int = 6

# The supervisor outputs exactly this token when the team has nothing left to
# contribute. Checked by exact, case-insensitive match so a specialist name
# that merely appears in a sentence can't be mistaken for a routing decision.
DONE_TOKEN: str = "DONE"

# The team. Each entry is name -> (one-line role summary shown to the
# supervisor, full system prompt that gives the agent its persona). Keeping
# both in one dict means the supervisor's menu and the agents themselves
# can't drift apart.
SPECIALISTS: dict[str, tuple[str, str]] = {
    "researcher": (
        "gathers the key facts, context, and constraints the task depends on",
        "You are the Researcher on a small team. Surface the concrete facts, "
        "background, and constraints that the task hinges on. Give a short "
        "bulleted list of findings — no recommendations, no prose essay.",
    ),
    "analyst": (
        "weighs the options, trade-offs, and risks",
        "You are the Analyst on a small team. Using the team's notes so far, "
        "lay out the realistic options and their trade-offs, call out risks, "
        "and state which direction you would lean and why. Be concise.",
    ),
    "writer": (
        "drafts and revises the actual deliverable the task asks for",
        "You are the Writer on a small team. Using the team's notes so far, "
        "produce (or revise) the actual deliverable the task asks for. Output "
        "only that deliverable — no commentary about your process.",
    ),
}

# Sample goals for testing. Each is deliberately many-sided so the team has
# something worth dividing up rather than a one-sentence answer.
SAMPLE_GOALS: list[str] = [
    "Recommend whether a 10-person startup should build its internal analytics "
    "on a managed data warehouse or a self-hosted one, and give the reasoning.",
    "Write a short launch announcement for a new open-source CLI tool that "
    "converts Markdown files into a static documentation site.",
]


def _format_transcript(transcript: list[tuple[str, str]]) -> str:
    """Render the (speaker, message) pairs so far as a text block for prompts."""
    if not transcript:
        return "(nothing yet — the team has not spoken)"
    return "\n\n".join(f"[{speaker}]\n{message}" for speaker, message in transcript)


def build_supervisor_chain() -> Runnable[dict[str, str], str]:
    """Build the chain that picks which teammate acts next.

    Given the goal and the transcript so far, it returns a single specialist
    name from ``SPECIALISTS`` or ``DONE_TOKEN``. `_choose_next()` normalises
    whatever the model actually writes back into one of those.
    """
    roster = "\n".join(f"- {name}: {summary}" for name, (summary, _) in SPECIALISTS.items())
    supervisor_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Supervisor of a small team. Your teammates are:\n"
                f"{roster}\n\n"
                "Look at the goal and the transcript so far, then decide who "
                "should act next. Reply with exactly one teammate's name, or "
                f'exactly "{DONE_TOKEN}" if the deliverable is finished and no '
                "further contribution would improve it. Output only that one "
                "word — no explanation.",
            ),
            ("human", "Goal:\n{goal}\n\nTranscript so far:\n{transcript}"),
        ]
    )
    return supervisor_prompt | llm | StrOutputParser()


def build_specialist_chain(name: str) -> Runnable[dict[str, str], str]:
    """Build the chain for one named specialist.

    Every specialist sees the same inputs — the goal and the full transcript —
    and differs only in its system prompt (its persona), pulled from
    ``SPECIALISTS``.
    """
    _, system_prompt = SPECIALISTS[name]
    specialist_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Goal:\n{goal}\n\nTeam transcript so far:\n{transcript}"),
        ]
    )
    return specialist_prompt | llm | StrOutputParser()


def build_editor_chain() -> Runnable[dict[str, str], str]:
    """Build the deterministic final pass that compiles the team's work.

    This always runs once at the end, regardless of what the supervisor did,
    so a run has a predictable output shape even if the team wandered.
    """
    editor_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Editor. You are given a goal and the full "
                "transcript of a team that worked on it. Produce the single "
                "best final answer to the goal, drawing on the team's notes "
                "and using the Writer's draft as your starting point. Output "
                "only the final answer, with no preamble.",
            ),
            ("human", "Goal:\n{goal}\n\nTeam transcript:\n{transcript}"),
        ]
    )
    return editor_prompt | llm | StrOutputParser()


def _choose_next(raw: str) -> str | None:
    """Map the supervisor's raw reply to a specialist name, ``DONE_TOKEN``, or None.

    Returns None when the reply matches nothing we recognise; the caller
    treats that the same as ``DONE_TOKEN`` (stop) but logs it.
    """
    # Reduce to the first "word-ish" token, lower-cased: this strips stray
    # punctuation, quotes, and any trailing sentence the model tacked on
    # despite being asked for one word.
    token = raw.strip().strip("\"'`.,:;!?").split()[0].lower() if raw.strip() else ""
    if token == DONE_TOKEN.lower():
        return DONE_TOKEN
    if token in SPECIALISTS:
        return token
    return None


def run_team(goal: str) -> str:
    """Run the supervisor-routed team loop for a single goal.

    The chains are stateless and rebuilt per call; the loop's only state is
    ``transcript``, the growing list of (speaker, message) pairs that every
    later agent — and the final editor pass — is given as context.

    Returns:
        The editor's compiled final answer.
    """
    supervisor_chain = build_supervisor_chain()
    editor_chain = build_editor_chain()
    # Build each specialist chain once and reuse it across turns.
    specialist_chains = {name: build_specialist_chain(name) for name in SPECIALISTS}

    transcript: list[tuple[str, str]] = []

    for turn in range(1, MAX_TURNS + 1):
        decision = supervisor_chain.invoke(
            {"goal": goal, "transcript": _format_transcript(transcript)}
        )
        nxt = _choose_next(decision)
        logger.info("Turn %d: supervisor chose %r (raw: %r)", turn, nxt, decision.strip())

        if nxt is None:
            logger.warning("Unrecognised supervisor reply; ending the team loop")
            break
        if nxt == DONE_TOKEN:
            print(f"Supervisor: DONE after {turn - 1} contribution(s)\n")
            break

        result = specialist_chains[nxt].invoke(
            {"goal": goal, "transcript": _format_transcript(transcript)}
        ).strip()
        transcript.append((nxt, result))
        print(f"[{nxt}]\n{result}\n")
    else:
        logger.info("Reached MAX_TURNS (%d) without the supervisor saying DONE", MAX_TURNS)

    final_answer = editor_chain.invoke(
        {"goal": goal, "transcript": _format_transcript(transcript)}
    ).strip()
    logger.info("Editor compiled the final answer for goal: %s", goal)
    return final_answer


def handle_requests() -> None:
    """Main entry point: run the multi-agent team for all sample goals.

    Registered in the CLI (see src/examples/__main__.py `_run_multiagent` and
    src/app/cli.py `CMD_MULTIAGENT`), invoked by:
        uv run python -m src.examples multiagent
    """
    logger.info("Running the multi-agent example over %d sample goals", len(SAMPLE_GOALS))
    print("Running supervisor-routed multi-agent team over sample goals")

    # Each goal is an independent run — run_team() starts from an empty
    # transcript every time.
    for goal in SAMPLE_GOALS:
        print("=" * 60)
        print(f"Goal: {goal}\n")
        final_answer = run_team(goal)
        print("-" * 60)
        print(f"Final answer:\n{final_answer}")
        print("=" * 60)
        print()
