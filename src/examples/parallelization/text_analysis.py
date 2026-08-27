"""Parallelization example using AWS Bedrock and LangChain.

Demonstrates the Parallelization agentic pattern: several independent LLM
sub-tasks (summary, sentiment, keyword extraction) run concurrently against
the same input via RunnableParallel, instead of one after another, and their
results are merged into a single report keyed by sub-task name.

The sub-tasks are safe to run in parallel precisely because none of them
consumes another's output — they only share the input text. Wall-clock time
for the whole analysis is therefore the slowest single branch, not the sum
of all three, and analyze_text() times the call to make that visible.

How this differs from the other patterns in this repo:
  - prompt_chaining: step 2 depends on step 1, so its calls must be
    sequential; here the branches are independent.
  - routing: exactly one branch runs, chosen by the model; here every branch
    runs and no choice is made.
  - reflection / planning / tools / multiagent: later steps depend on earlier
    results, so they cannot be collapsed into one parallel fan-out.

Flow for one text:

    text ── RunnableParallel(summary=..., sentiment=..., keywords=...)
         ── all three sub-chains dispatched at once (thread pool)
         ── {"summary": "...", "sentiment": "...", "keywords": "..."}
"""

import logging
import time
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel
from ...app.bedrock import llm

logger = logging.getLogger(__name__)

# Sample texts for testing
SAMPLE_TEXTS: list[str] = [
    "The new smartphone launch exceeded sales expectations, with customers praising "
    "the battery life and camera quality, though several reviewers noted the price "
    "is steep compared to competing models.",
    "The city council approved the new public transit plan after months of debate. "
    "Residents are divided, with some welcoming reduced traffic congestion and "
    "others worried about construction noise and disruption during the build-out.",
]


def build_parallel_chain() -> Runnable[dict[str, Any], dict[str, str]]:
    """Build a chain that runs three independent analyses of the same text concurrently.

    Each sub-task (summary, sentiment, keywords) is its own linear chain:
    prompt | llm | StrOutputParser(). They don't depend on each other's
    output, which is exactly what makes them safe to fan out via
    RunnableParallel instead of running them one after another.

    Returns:
        A Runnable that takes {"text": ...} and returns a dict with
        "summary", "sentiment", and "keywords" keys.
    """
    # Each prompt is deliberately constrained to a narrow, easy-to-parse
    # output shape (one sentence / one word / a comma list) since these
    # results are merged programmatically afterwards, not read by another
    # LLM step.
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one concise sentence:\n\n{text}"
    )
    sentiment_prompt = ChatPromptTemplate.from_template(
        "Classify the overall sentiment of the following text as exactly one word "
        "(positive, negative, or neutral). Output only that word:\n\n{text}"
    )
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract the 3-5 most important keywords from the following text as a "
        "comma-separated list. Output only the list:\n\n{text}"
    )

    # Each is a standalone LCEL chain that could be invoked on its own;
    # RunnableParallel below is what turns them into concurrent branches
    # rather than a shared dependency.
    summary_chain = summary_prompt | llm | StrOutputParser()
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    keywords_chain = keywords_prompt | llm | StrOutputParser()

    # RunnableParallel dispatches all three sub-chains against the same input
    # and runs them concurrently (via a thread pool under the hood), rather
    # than waiting for one to finish before starting the next. The keyword
    # arguments here (summary=, sentiment=, keywords=) become the keys of the
    # result dict returned by .invoke(), so each branch's output lands under
    # its own name without any manual merging step.
    analysis_chain = RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
        keywords=keywords_chain,
    )

    logger.debug("Parallel analysis chain built successfully")
    return analysis_chain


def analyze_text(chain: Runnable[dict[str, Any], dict[str, str]], text: str) -> None:
    """Run the parallel analysis chain on a single text and print the merged report."""
    print("Text:", text)
    print()

    # A single .invoke() call triggers all three sub-chains at once; the
    # elapsed time reflects the slowest branch, not the sum of all three,
    # which is the concrete payoff of running them in parallel rather than
    # sequentially (summary_chain.invoke(), then sentiment_chain.invoke(),
    # then keywords_chain.invoke()).
    start = time.perf_counter()
    result = chain.invoke({"text": text})
    elapsed = time.perf_counter() - start

    # result is a dict keyed by the branch names passed to RunnableParallel
    # (summary/sentiment/keywords); .strip() trims incidental whitespace
    # some models add around their output.
    print(f"Summary:   {result['summary'].strip()}")
    print(f"Sentiment: {result['sentiment'].strip()}")
    print(f"Keywords:  {result['keywords'].strip()}")
    print(f"(3 sub-tasks completed concurrently in {elapsed:.2f}s)")
    logger.info("Parallel analysis completed in %.2fs", elapsed)
    print("-" * 60)


def handle_requests() -> None:
    """Main entry point: build the parallel chain and analyze all sample texts."""
    logger.info("Building parallel text analysis chain")
    print("Building parallel text analysis chain")
    # The chain is built once and reused across all sample texts, since
    # RunnableParallel/the sub-chains carry no per-invocation state.
    analysis_chain = build_parallel_chain()
    for text in SAMPLE_TEXTS:
        analyze_text(analysis_chain, text)
