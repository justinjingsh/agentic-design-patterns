"""Goal Setting and Monitoring example using AWS Bedrock and LangChain.

Demonstrates the **Goal Setting and Monitoring** agentic pattern: give the
agent a high-level goal, have it turn that goal into an explicit checklist of
*measurable success criteria*, then let it work toward the goal in a loop
where after every attempt it **monitors** the current result against each
criterion and decides whether to keep going. The loop stops as soon as every
criterion is met (success) or a hard iteration cap is hit (give up and report
what still fails).

The two halves of the pattern:

  - **Goal setting** — ``build_criteria_chain()`` decomposes a fuzzy goal
    ("write a launch email that...") into a numbered list of concrete,
    individually checkable statements ("states the $299 price", "under 120
    words"). These criteria, not the original prose, are what "done" means
    for the rest of the run.
  - **Monitoring** — ``build_monitor_chain()`` scores the current draft
    against that checklist, returning ``MET`` / ``UNMET`` plus a one-line
    reason per criterion. ``_parse_progress()`` turns that into structured
    data the loop can branch on, and the still-``UNMET`` items (with their
    reasons) are fed back into the next attempt as targeted feedback.

Per-goal flow:

    goal ── criteria_chain ──> [criterion, criterion, ...]     (goal setting)
         ── loop, up to MAX_ITERATIONS times:
              worker_chain(goal, criteria, draft, feedback) ──> new draft
              monitor_chain(criteria, draft) ──> per-criterion MET/UNMET + why
                all MET?  ── yes ──> stop, success
                          ── no  ──> feedback = the UNMET lines; loop again
         ── return the final draft + the last progress report

How this differs from the neighbouring patterns in this repo:
  - planning: commits to a fixed ordered step list up front and executes it
    once. Here there is no step list — the agent re-attempts the *whole*
    task each iteration, steered only by which criteria still fail.
  - reflection: a free-form critic critiques the draft in prose. Here the
    "critic" is pinned to a fixed, goal-derived checklist, so progress is a
    countable "3 of 5 criteria met", not a vibe — that measurability is the
    whole point of the pattern.
  - tools / multiagent: the model chooses what happens next. Here the
    control flow is fixed; only the number of iterations varies, gated by an
    objective stop condition.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# The most success criteria to derive from a single goal. Keeping the
# checklist short keeps each monitoring pass cheap and its verdict easy to
# parse; a real system would tune this to the goal's complexity.
MAX_CRITERIA: int = 6

# Hard ceiling on work/monitor rounds so a goal the model can't quite satisfy
# (or a monitor that keeps moving the goalposts) can't loop forever. On hit,
# the loop returns the best draft so far plus the unmet criteria.
MAX_ITERATIONS: int = 4

# Tokens the monitor must start each verdict line with. Checked by exact,
# upper-cased match in `_parse_progress` so a reason that happens to contain
# the word "met" doesn't flip the verdict.
MET_TOKEN: str = "MET"
UNMET_TOKEN: str = "UNMET"

# Goals written so their criteria are objectively checkable (word counts,
# required substrings, a sign-off) — that makes the monitoring loop visibly
# converge instead of arguing about taste.
SAMPLE_GOALS: list[str] = [
    (
        "Write a launch announcement email for the ADP-500 wireless headphones. "
        "It must have a subject line, the body must be under 60 words, it must "
        "state the $299 price and the 40-hour battery life, it must contain no "
        "exclamation marks, it must include a call to action that starts with "
        "the word 'Reserve', and it must end with the exact sign-off line "
        "'- The ADP Team'."
    ),
    (
        "Draft a short onboarding message for new users of a note-taking app "
        "named Jot. It must greet the user as '[name]', name exactly two "
        "features (quick capture and tags), contain the word 'today', include a "
        "'[getting-started guide]' link placeholder, and be no more than 45 words."
    ),
]


def build_criteria_chain() -> Runnable[dict[str, str], str]:
    """Build the goal-setting chain: fuzzy goal -> checkable criteria list.

    Runs once per goal, before the loop. Returns a newline-separated numbered
    list; ``_parse_criteria`` splits it into a ``list[str]``. The prompt
    pushes hard for *atomic, verifiable* items because every criterion here
    becomes a yes/no question the monitor must answer later.
    """
    criteria_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You turn a goal into a checklist of success criteria. Output at most "
                f"{MAX_CRITERIA} criteria, one per line, numbered '1.', '2.', ... . "
                "Each must be atomic (one testable idea) and objectively verifiable by "
                "reading the finished work — prefer concrete thresholds and required "
                "content over vague quality words. Derive criteria ONLY from what the "
                "goal explicitly states; do not invent extra constraints, and quote any "
                "required exact string or placeholder from the goal verbatim. Output "
                "only the list.",
            ),
            ("human", "Goal:\n{goal}"),
        ]
    )
    return criteria_prompt | llm | StrOutputParser()


def build_worker_chain() -> Runnable[dict[str, str], str]:
    """Build the chain that produces (or revises) the work product.

    Takes the goal, the full criteria checklist, the current ``draft`` (empty
    on the first pass), and ``feedback`` — the monitor's list of still-unmet
    criteria and why. It re-does the whole task each call rather than editing
    in place, so a fix for one criterion can freely rework the rest.
    """
    worker_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You produce work that satisfies every listed success criterion. You "
                "are given the criteria, your previous draft (may be empty), and "
                "feedback on which criteria are not yet met. Return a complete new "
                "version that satisfies all criteria. Output only the work itself, "
                "with no commentary.",
            ),
            (
                "human",
                "Goal:\n{goal}\n\nSuccess criteria:\n{criteria}\n\n"
                "Previous draft:\n{draft}\n\nFeedback (unmet criteria):\n{feedback}",
            ),
        ]
    )
    return worker_prompt | llm | StrOutputParser()


def build_monitor_chain() -> Runnable[dict[str, str], str]:
    """Build the monitoring chain: score a draft against the checklist.

    Takes the numbered ``criteria`` and the ``draft`` and must return exactly
    one line per criterion, in order, shaped ``<n>. MET|UNMET - <reason>``.
    That rigid shape is what lets ``_parse_progress`` read the verdict back
    out deterministically — the reason is for the human and for the worker's
    next-round feedback.
    """
    monitor_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict monitor. For each numbered success criterion, judge "
                "ONLY whether the draft below satisfies it. When a criterion names a "
                "number (word count, sentence count, count of items, a required exact "
                "string), actually count or search for it and be conservative — mark "
                f"{MET_TOKEN} only when the draft clearly satisfies the criterion, "
                f"otherwise {UNMET_TOKEN}. Output exactly one line per criterion, in the "
                f"same order, formatted as '<number>. {MET_TOKEN}' or "
                f"'<number>. {UNMET_TOKEN} - <short reason>'. Do not add any other text.",
            ),
            ("human", "Success criteria:\n{criteria}\n\nDraft:\n{draft}"),
        ]
    )
    return monitor_prompt | llm | StrOutputParser()


def _parse_criteria(raw: str) -> list[str]:
    """Normalise the criteria chain's numbered list into a ``list[str]``.

    Lenient like the planning example's ``_parse_plan``: strips ``1.`` /
    ``1)`` / ``- `` / ``* `` prefixes, drops blank lines, and truncates to
    ``MAX_CRITERIA`` rather than raising if the model over-produces.
    """
    criteria: list[str] = []
    for line in raw.splitlines():
        text = line.strip().lstrip("-*0123456789.)( ").strip()
        if text:
            criteria.append(text)
    return criteria[:MAX_CRITERIA]


def _parse_progress(raw: str, criteria: list[str]) -> list[tuple[str, bool, str]]:
    """Turn the monitor's lines into ``(criterion, met, reason)`` triples.

    Matching is positional: the Nth non-empty verdict line pairs with the Nth
    criterion, so a stray/missing line doesn't silently shift every verdict.
    Any criterion without a parseable ``MET`` line is treated as UNMET — the
    safe default, since "couldn't confirm" must not end the loop early.
    """
    verdicts: list[tuple[bool, str]] = []
    for line in raw.splitlines():
        stripped = line.strip().lstrip("0123456789.)( ").strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith(UNMET_TOKEN):
            reason = stripped[len(UNMET_TOKEN):].lstrip(" -–:").strip()
            verdicts.append((False, reason or "not satisfied"))
        elif upper.startswith(MET_TOKEN):
            verdicts.append((True, ""))
        # Lines that match neither token are noise; skip them.

    progress: list[tuple[str, bool, str]] = []
    for index, criterion in enumerate(criteria):
        if index < len(verdicts):
            met, reason = verdicts[index]
        else:
            met, reason = False, "no verdict returned"
        progress.append((criterion, met, reason))
    return progress


def _format_criteria(criteria: list[str]) -> str:
    """Render the checklist as a numbered block for the prompts."""
    return "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))


def _format_feedback(progress: list[tuple[str, bool, str]]) -> str:
    """Render just the unmet criteria (with reasons) for the worker's retry."""
    unmet = [f"- {c} (monitor: {reason})" for c, met, reason in progress if not met]
    return "\n".join(unmet) if unmet else "(none — all criteria met)"


def run_goal_loop(goal: str) -> tuple[str, list[tuple[str, bool, str]]]:
    """Run goal setting + the monitored work loop for one goal.

    Returns:
        ``(final_draft, final_progress)`` — the last work product and the
        monitor's last per-criterion verdict list, whether the loop exited on
        success or on hitting ``MAX_ITERATIONS``.
    """
    criteria_chain = build_criteria_chain()
    worker_chain = build_worker_chain()
    monitor_chain = build_monitor_chain()

    # --- Goal setting: done once, up front. ---
    criteria = _parse_criteria(criteria_chain.invoke({"goal": goal}))
    criteria_block = _format_criteria(criteria)
    print(f"Success criteria ({len(criteria)}):\n{criteria_block}\n")
    logger.info("Derived %d success criteria for goal", len(criteria))

    # --- Loop state: the evolving draft + the latest monitor report. ---
    draft = ""
    progress: list[tuple[str, bool, str]] = []
    feedback = "(first attempt — no feedback yet)"

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Act: (re)produce the whole work product, steered by the feedback.
        draft = worker_chain.invoke(
            {"goal": goal, "criteria": criteria_block, "draft": draft or "(none)", "feedback": feedback}
        ).strip()
        print(f"--- Attempt {iteration} ---\n{draft}\n")

        # Monitor: score the fresh draft against every criterion.
        progress = _parse_progress(
            monitor_chain.invoke({"criteria": criteria_block, "draft": draft}), criteria
        )
        met_count = sum(1 for _, met, _ in progress if met)
        print(f"Monitor: {met_count}/{len(criteria)} criteria met")
        for crit, met, reason in progress:
            mark = "OK  " if met else "MISS"
            print(f"  [{mark}] {crit}" + (f"  <- {reason}" if reason and not met else ""))
        print()
        logger.info("Iteration %d: %d/%d criteria met", iteration, met_count, len(criteria))

        # Check the objective stop condition.
        if met_count == len(criteria) and criteria:
            logger.info("All criteria met after %d iteration(s)", iteration)
            print(f"Goal reached after {iteration} iteration(s).\n")
            break

        # Otherwise turn the unmet criteria into the next attempt's feedback.
        feedback = _format_feedback(progress)
    else:
        logger.info("Hit MAX_ITERATIONS (%d) with unmet criteria", MAX_ITERATIONS)
        print(f"Stopped after {MAX_ITERATIONS} iterations with criteria still unmet.\n")

    return draft, progress


def handle_requests() -> None:
    """Main entry point: run the goal loop over every sample goal."""
    logger.info("Running goal setting and monitoring over %d goals", len(SAMPLE_GOALS))

    for goal in SAMPLE_GOALS:
        print("=" * 60)
        print(f"Goal: {goal}\n")
        final_draft, final_progress = run_goal_loop(goal)
        met = sum(1 for _, ok, _ in final_progress if ok)
        print("-" * 60)
        print(f"Final result ({met}/{len(final_progress)} criteria met):\n{final_draft}")
        print("=" * 60)
        print()
