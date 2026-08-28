"""Routing example using AWS Bedrock and LangChain.

Demonstrates the Routing agentic pattern: an LLM classifies each incoming
request into one of a fixed set of categories, and a RunnableBranch then
dispatches the *untouched* request to the plain-Python handler registered
for that category. The model's only job is the one classification decision;
everything after it is deterministic.

How this differs from the other patterns in this repo:
  - prompt_chaining: a fixed two-step `|` chain — no decision is made, the
    steps are always the same.
  - parallelization: every sub-task runs; here exactly one handler runs and
    the model picks which.
  - reflection / planning / tools / multiagent: the model keeps making
    decisions across multiple turns; here it decides once and is then out of
    the loop.

Flow for one request:

    request ── {category: classifier_chain, request: passthrough}
            ── produces {"category": "...", "request": "<original text>"}
            ── RunnableBranch: first matching category wins, else default
            ── handler(request)  ──> result string
"""

import logging
from typing import Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from ....app.bedrock import llm


logger = logging.getLogger(__name__)

# Sample requests for testing
TEST_REQUESTS: list[str] = [
    "book a flight to New York",
    "what is the capital of Italy?",
    "I need to book a hotel room in Tokyo",
    "Can you tell me the weather in Paris?",
    "I want to book a table at a restaurant",
    "random text that doesn't match any handler",
]


# --- Handlers -------------------------------------------------------------
# Each handler only cares about the raw request text, not how it was
# selected — routing and handling are fully decoupled. In a real system
# these would call booking APIs, a knowledge base, etc.; here they return a
# canned string so the routing behaviour is what the example shows.

def booking_handler(request: str) -> str:
    """Handle any reservation/booking request (flight, hotel, table, ticket)."""
    logger.info("Handling booking request: %s", request)
    # Implement booking logic here
    return "Booking confirmed"


def info_handler(request: str) -> str:
    """Handle any information-lookup request (weather, facts, locations)."""
    logger.info("Handling info request: %s", request)
    # Implement information retrieval logic here
    return "Information provided"


def unclear_handler(request: str) -> str:
    """Fallback for requests the classifier couldn't place in another category."""
    logger.info("Handling unclear request: %s", request)
    # Implement logic for unclear requests here
    return "Request is unclear, please provide more details"


# Single source of truth mapping a classifier category to its handler.
# build_router() derives the RunnableBranch conditions from this dict, so
# adding a new category only means adding one entry here — the category
# string is never duplicated elsewhere.
HANDLERS: dict[str, Callable[[str], str]] = {
    "booker": booking_handler,
    "info": info_handler,
}
DEFAULT_HANDLER: Callable[[str], str] = unclear_handler


def build_router() -> Runnable[dict[str, str], str]:
    """Build a router chain that classifies a request and dispatches it to the matching handler."""
    logger.info("Building router with handlers")

    # Prompt instructs the LLM to act purely as a classifier: it must return
    # exactly one of the HANDLERS keys and nothing else, so the output can be
    # used directly as a routing key without extra parsing.
    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Analyze the user request and output only one category identifier: booker, info, or unclear.\n"
                "booker: any reservation/booking request, e.g. booking a flight, hotel, restaurant table, or ticket.\n"
                "info: any information request, e.g. asking about weather, locations, or facts.\n"
                "unclear: any request that does not fit the above two categories.\n"
                "Do not output any extra text.",
            ),
            ("human", "{request}"),
        ]
    )

    # Classifier pipeline (LCEL): format prompt -> call LLM -> extract the
    # plain string content -> strip stray whitespace/newlines so the result
    # is an exact match for one of the HANDLERS keys.
    coordinator_router_chain = router_prompt | llm | StrOutputParser() | (lambda s: s.strip())

    # RunnableBranch behaves like if/elif/.../else: each (condition, runnable)
    # pair is tested in order, the first matching condition's runnable runs
    # and short-circuits, and the final "bare" runnable (no condition) is the
    # default fallback when no category matches.
    # Default args (category=category, handler=handler) bind each lambda's
    # loop variables at definition time, avoiding the classic late-binding
    # closure bug where every branch would end up referencing the last item.
    branch = RunnableBranch(
        *(
            (
                lambda x, category=category: x["category"] == category,
                RunnableLambda(lambda x, handler=handler: handler(x["request"])),
            )
            for category, handler in HANDLERS.items()
        ),
        RunnableLambda(lambda x: DEFAULT_HANDLER(x["request"])),
    )

    # Dict-of-runnables (RunnableParallel shorthand): both entries receive the
    # SAME original input {"request": ...}. "category" runs the classifier on
    # it, while "request" is an identity passthrough that carries the original
    # text forward unchanged. This produces {"category": ..., "request": ...},
    # which then feeds into `branch` so the chosen handler still gets the
    # original, untouched request rather than the classification result.
    coordinator_agent = (
        {"category": coordinator_router_chain, "request": lambda x: x["request"]}
        | branch
    )
    return coordinator_agent


def handle_requests() -> None:
    """Main entry point: build the router and process all sample requests."""
    logger.info("Handle user specific request and route to different task")
    coordinator_agent = build_router()

    for request in TEST_REQUESTS:
        logger.info("Processing request: %s", request)
        # A single .invoke() runs the whole pipeline: classify -> route -> handle.
        result = coordinator_agent.invoke({"request": request})
        print(f"Request: {request}")
        print(f"Result: {result}")
        print("-" * 60)
