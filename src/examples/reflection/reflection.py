"""Reflection example using AWS Bedrock and LangChain.

Demonstrates the Reflection agentic pattern: an LLM produces a first-draft
answer (the "generator"), a second LLM pass critiques that draft against the
original task (the "reflector"), and — unless the critique approves the draft
outright — a third pass revises the draft using the critique (the "refiner").
This generate -> reflect -> refine loop repeats up to a fixed number of
iterations, giving the model a chance to catch and fix its own mistakes
instead of returning the first thing it produced.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from ...app.bedrock import llm

logger = logging.getLogger(__name__)

# Reflection stops early once the reflector approves a draft, but is capped
# here so a stubborn reflector (or a task with no clean "approved" state)
# can't loop forever.
MAX_ITERATIONS: int = 3

# The reflector outputs exactly this token when it has no more notes; the
# refine step is skipped once it appears, so it must not collide with any
# critique the model would plausibly write.
APPROVAL_TOKEN: str = "APPROVED"

# Sample tasks for testing
SAMPLE_TASKS: list[str] = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a one-paragraph product description for a reusable water bottle.",
]


def build_generate_chain() -> Runnable[dict[str, str], str]:
    """Build the chain that produces the first-draft answer to a task."""
    generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You complete the given task as well as you can in a single attempt. "
                "Output only the result, with no preamble or explanation.",
            ),
            ("human", "{task}"),
        ]
    )
    return generate_prompt | llm | StrOutputParser()


def build_reflect_chain() -> Runnable[dict[str, str], str]:
    """Build the chain that critiques a draft against the original task.

    The reflector never rewrites the draft itself — it only ever judges it —
    which keeps its role distinct from the refine step below.
    """
    reflect_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict reviewer. Given a task and a draft response to it, "
                "look for correctness bugs, missed requirements, and unclear or "
                f"low-quality writing. If the draft is already correct and complete, "
                f'respond with exactly "{APPROVAL_TOKEN}" and nothing else. Otherwise, '
                "output a short, concrete list of the problems to fix — do not rewrite "
                "the draft yourself.",
            ),
            ("human", "Task:\n{task}\n\nDraft:\n{draft}"),
        ]
    )
    return reflect_prompt | llm | StrOutputParser()


def build_refine_chain() -> Runnable[dict[str, str], str]:
    """Build the chain that rewrites a draft using the reflector's critique."""
    refine_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You revise a draft response to address reviewer feedback. Apply the "
                "feedback fully while keeping everything about the draft that the "
                "feedback didn't flag. Output only the revised result, with no preamble "
                "or explanation.",
            ),
            (
                "human",
                "Task:\n{task}\n\nDraft:\n{draft}\n\nReviewer feedback:\n{critique}",
            ),
        ]
    )
    return refine_prompt | llm | StrOutputParser()


def run_reflection_loop(task: str) -> str:
    """Run the generate -> reflect -> refine loop for a single task.

    Each chain is stateless, so it's built fresh per call rather than shared
    across iterations — the loop's only state is the evolving `draft` string.

    Returns:
        The final draft: either the one the reflector approved, or the draft
        from the last iteration if MAX_ITERATIONS was reached first.
    """
    generate_chain = build_generate_chain()
    reflect_chain = build_reflect_chain()
    refine_chain = build_refine_chain()

    draft = generate_chain.invoke({"task": task})
    print(f"Draft 1:\n{draft.strip()}\n")
    logger.info("Generated initial draft for task: %s", task)

    for iteration in range(1, MAX_ITERATIONS + 1):
        critique = reflect_chain.invoke({"task": task, "draft": draft}).strip()
        print(f"Critique {iteration}: {critique}\n")
        logger.info("Reflection pass %d critique: %s", iteration, critique)

        # Exact-match on the approval token (rather than a substring check)
        # avoids false positives from a critique that merely mentions the
        # word while still listing problems.
        if critique == APPROVAL_TOKEN:
            logger.info("Draft approved after %d reflection pass(es)", iteration)
            break

        draft = refine_chain.invoke({"task": task, "draft": draft, "critique": critique})
        print(f"Draft {iteration + 1}:\n{draft.strip()}\n")
        logger.info("Refined draft after reflection pass %d", iteration)
    else:
        logger.info("Reached MAX_ITERATIONS (%d) without approval", MAX_ITERATIONS)

    return draft


def handle_requests() -> None:
    """Main entry point: run the reflection loop for all sample tasks."""
    logger.info("Building reflection chain")
    print("Running reflection loop over sample tasks")

    for task in SAMPLE_TASKS:
        print("=" * 60)
        print(f"Task: {task}\n")
        final_draft = run_reflection_loop(task)
        print("-" * 60)
        print(f"Final result:\n{final_draft.strip()}")
        print("=" * 60)
        print()
