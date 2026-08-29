"""Retrieval-Augmented Generation (RAG) example using AWS Bedrock and LangChain.

Demonstrates the **RAG** (a.k.a. Knowledge Retrieval) agentic pattern: before
the model answers, a deterministic retrieval step pulls the most relevant
passages out of a knowledge base and injects them into the prompt, and the
model is told to answer *only* from those passages and to cite them.

Placed under ``reliability_layers/`` because that grounding is a reliability
property, not just a convenience:

  - **Faithfulness.** The answer is tied to supplied source text, so the model
    is far less free to hallucinate a plausible-sounding wrong fact.
  - **Attribution.** Every claim carries a ``[n]`` pointing at the passage it
    came from, so a reader can check it.
  - **Bounded knowledge / honest refusal.** When retrieval surfaces nothing
    that answers the question, the agent says so (``INSUFFICIENT_CONTEXT``)
    instead of guessing from parametric memory.
  - **Freshness.** Swapping documents in the store changes the answers with no
    retraining — the knowledge lives outside the weights.

Fixed three-stage pipeline (the model makes no control-flow decisions):

    question
      1. RETRIEVE  score every document against the query, keep the top
                   MAX_CONTEXT_DOCS whose score clears MIN_RETRIEVAL_SCORE
      2. AUGMENT   render those passages as a numbered SOURCES block
      3. GENERATE  answer_chain(sources, question) -> grounded, cited answer
                   (or exactly INSUFFICIENT_CONTEXT)
    then a deterministic post-check flags any [n] citation that points
    outside the retrieved set.

Simplifications that keep the example offline and deterministic:
  - The retriever is plain lexical overlap (query terms n document terms),
    not embeddings + a vector store. Production RAG swaps in a semantic
    retriever, but the retrieve -> augment -> generate shape is identical.
  - The knowledge base is a hard-coded list of short documents.

How this differs from the neighbouring patterns in this repo:
  - tools: there the *model* decides at runtime to call a data-fetching
    function and which arguments to pass. Here retrieval is a fixed step that
    always runs first and is not the model's decision.
  - exception_handling / hitl: those wrap each tool call to recover from a
    failure / gate a side effect. RAG has no tool loop at all — it grounds a
    single-shot answer and declines when the grounding is missing.
  - reflection / goal_monitoring: those improve a weak answer by iterating.
    RAG improves the answer by changing what the model is *given* before it
    writes anything.
"""

import logging
import re
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# Shared ChatBedrock instance, constructed once at import time in
# src/app/bedrock.py (which also validates AWS credentials at module scope).
from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# Top-k: how many retrieved passages are placed in the prompt. Kept small so
# the retriever must actually pick the right ones rather than dumping the
# whole corpus into context.
MAX_CONTEXT_DOCS: int = 3

# A document must share at least this many query terms to count as relevant at
# all. Guards the "nothing matched" path: a question about a topic the corpus
# doesn't cover retrieves zero passages and short-circuits to a refusal
# without an LLM call.
MIN_RETRIEVAL_SCORE: int = 1

# The generator emits exactly this string (and nothing else) when the SOURCES
# block does not contain the answer. Checked by exact match, like the
# APPROVAL_TOKEN in the reflection example, so a passing mention doesn't count.
INSUFFICIENT_CONTEXT_TOKEN: str = "INSUFFICIENT_CONTEXT"

# Words ignored when scoring lexical overlap, so matches are driven by content
# terms rather than glue words.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have how in is it its of on or that
    the to was what when where which who why with you your do does can
    """.split()
)


@dataclass(frozen=True)
class Document:
    """One entry in the knowledge base."""

    doc_id: str
    title: str
    text: str


# A small, self-contained knowledge base: the product documentation for a
# fictional service. Facts are spread across documents on purpose so that
# answering some questions needs more than one passage, and one sample
# question asks about something deliberately absent.
KNOWLEDGE_BASE: list[Document] = [
    Document(
        "kb-pricing",
        "CloudSync pricing tiers",
        "CloudSync has three plans. Free covers a single user and 2 GB of "
        "storage. Team is 12 USD per user per month and includes 1 TB of "
        "pooled storage. Enterprise is custom-priced and adds SSO, audit "
        "logs, and a dedicated support contact.",
    ),
    Document(
        "kb-retention",
        "Backup and data retention",
        "CloudSync keeps deleted files in a recycle bin for 30 days on the "
        "Free and Team plans and 90 days on Enterprise. Point-in-time backup "
        "snapshots are taken every 6 hours and retained for 35 days.",
    ),
    Document(
        "kb-regions",
        "Supported storage regions",
        "Customer data can be pinned to one of four regions: US East, EU "
        "(Frankfurt), UK (London), or Asia Pacific (Sydney). The region is "
        "chosen at account creation and cannot be changed later without "
        "contacting support.",
    ),
    Document(
        "kb-sla",
        "Service level agreement",
        "The CloudSync SLA guarantees 99.9% monthly uptime for the Team plan "
        "and 99.95% for Enterprise. If uptime falls below the guarantee, "
        "affected customers receive service credits of 10% of the monthly fee "
        "per half-percent missed.",
    ),
    Document(
        "kb-security",
        "Encryption and security",
        "All CloudSync data is encrypted in transit with TLS 1.2 or higher. "
        "Data at rest is encrypted with AES-256. Enterprise customers may "
        "supply their own KMS key for envelope encryption. Passwords are "
        "hashed with bcrypt.",
    ),
    Document(
        "kb-support",
        "Support channels and hours",
        "Free plan support is community forum only. Team plan includes email "
        "support with a one business day response target. Enterprise adds 24/7 "
        "phone support and a one-hour response target for critical issues.",
    ),
]

# Each question targets a different retrieval outcome.
SAMPLE_QUESTIONS: list[str] = [
    # Single-document answer: everything needed is in kb-sla.
    "What uptime does the SLA guarantee for the Team plan?",
    # Multi-document synthesis: needs kb-security (AES-256) and kb-retention
    # (35-day snapshot retention) together, so the answer should cite both.
    "How is data at rest encrypted, and how long are backup snapshots kept?",
    # Out of scope: the corpus says nothing about a mobile app, so retrieval
    # is weak / irrelevant and the agent should return INSUFFICIENT_CONTEXT
    # rather than inventing an answer.
    "Is there a native iOS mobile app, and does it support offline editing?",
]


def _tokenize(text: str) -> set[str]:
    """Lower-case alphanumeric word set with stopwords and 1-char tokens removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _score(query_terms: set[str], doc: Document) -> int:
    """Relevance score: how many distinct query terms the document contains.

    Title terms are counted twice — a query word that appears in the
    document's title is a strong topical signal. This is a crude stand-in for
    the cosine similarity an embedding retriever would compute; it is fully
    deterministic, which keeps the example reproducible.
    """
    body_hits = query_terms & _tokenize(doc.text)
    title_hits = query_terms & _tokenize(doc.title)
    return len(body_hits) + len(title_hits)


def retrieve(question: str) -> list[tuple[Document, int]]:
    """Stage 1: return the most relevant documents, best first.

    Keeps at most ``MAX_CONTEXT_DOCS`` and drops anything scoring below
    ``MIN_RETRIEVAL_SCORE`` so an off-topic question can legitimately retrieve
    nothing. Ties break on ``doc_id`` for a stable order.
    """
    query_terms = _tokenize(question)
    scored = ((doc, _score(query_terms, doc)) for doc in KNOWLEDGE_BASE)
    ranked = sorted(
        (pair for pair in scored if pair[1] >= MIN_RETRIEVAL_SCORE),
        key=lambda pair: (-pair[1], pair[0].doc_id),
    )
    return ranked[:MAX_CONTEXT_DOCS]


def _format_sources(retrieved: list[tuple[Document, int]]) -> str:
    """Stage 2: render retrieved passages as a numbered SOURCES block.

    The ``[n]`` labels here are what the model is asked to cite, and what
    ``_check_citations`` later validates against.
    """
    return "\n\n".join(
        f"[{index}] {doc.title}\n{doc.text}"
        for index, (doc, _score_value) in enumerate(retrieved, start=1)
    )


def build_answer_chain() -> Runnable[dict[str, str], str]:
    """Stage 3: the grounded-answer chain (``prompt | llm | parser``).

    The system prompt pins three rules: answer only from SOURCES, cite every
    claim with its ``[n]``, and fall back to ``INSUFFICIENT_CONTEXT_TOKEN``
    (exact string, nothing else) when the sources don't hold the answer.
    Stateless and rebuilt per run, like every other chain in this repo.
    """
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions about the CloudSync product using ONLY the "
                "numbered SOURCES provided. Follow these rules exactly:\n"
                "1. Base every statement on the SOURCES. Do not use outside "
                "knowledge or guess.\n"
                "2. After each claim, cite the source it came from as [n], "
                "matching the numbers in the SOURCES block. Combine sources when "
                "a claim needs more than one.\n"
                f"3. If the SOURCES do not contain enough information to answer, "
                f"reply with exactly '{INSUFFICIENT_CONTEXT_TOKEN}' and nothing "
                "else.\n"
                "Keep the answer to a few sentences.",
            ),
            (
                "human",
                "SOURCES:\n{sources}\n\nQUESTION: {question}",
            ),
        ]
    )
    return answer_prompt | llm | StrOutputParser()


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _check_citations(answer: str, num_sources: int) -> list[int]:
    """Return any cited source numbers that don't exist in the retrieved set.

    A non-empty result means the model referenced a ``[n]`` we never gave it —
    a grounding smell worth surfacing even though we don't hard-fail on it
    (the parsing stays lenient, like ``_parse_plan`` in the planning example).
    """
    cited = {int(n) for n in _CITATION_RE.findall(answer)}
    valid = set(range(1, num_sources + 1))
    return sorted(cited - valid)


def run_rag(question: str, answer_chain: Runnable[dict[str, str], str]) -> str:
    """Run the full retrieve -> augment -> generate pipeline for one question."""
    retrieved = retrieve(question)

    if not retrieved:
        # Stage 1 found nothing above the relevance floor: don't even call the
        # model — there is nothing to ground an answer in.
        print("  [retrieved] nothing above the relevance threshold")
        logger.info("No documents retrieved for question: %s", question)
        return (
            f"{INSUFFICIENT_CONTEXT_TOKEN} - the knowledge base has no documents "
            "relevant to this question."
        )

    for index, (doc, score_value) in enumerate(retrieved, start=1):
        print(f"  [retrieved] [{index}] {doc.doc_id} (score {score_value}): {doc.title}")

    sources = _format_sources(retrieved)
    answer = answer_chain.invoke({"sources": sources, "question": question}).strip()

    if answer == INSUFFICIENT_CONTEXT_TOKEN:
        logger.info("Model reported insufficient context for: %s", question)
        return (
            f"{INSUFFICIENT_CONTEXT_TOKEN} - the retrieved documents do not "
            "answer this question."
        )

    unknown = _check_citations(answer, len(retrieved))
    if unknown:
        logger.warning("Answer cites unknown source(s) %s: %s", unknown, answer)
        print(f"  [warn] answer cites source(s) not retrieved: {unknown}")

    used = ", ".join(doc.doc_id for doc, _score_value in retrieved)
    return f"{answer}\n\n(grounded in: {used})"


def handle_requests() -> None:
    """Main entry point: run the RAG pipeline over the sample questions.

    Registered in the CLI (see src/examples/__main__.py `_run_rag` and
    src/app/cli.py `CMD_RAG`), invoked by:
        uv run python -m src.examples rag
    """
    logger.info("Running RAG pipeline over %d questions", len(SAMPLE_QUESTIONS))
    print("=" * 60)
    print(f"RAG: retrieve -> augment -> generate over {len(KNOWLEDGE_BASE)} documents")
    print("=" * 60)

    answer_chain = build_answer_chain()

    for question in SAMPLE_QUESTIONS:
        print(f"\nQuestion: {question}")
        answer = run_rag(question, answer_chain)
        print("-" * 60)
        print(f"Answer: {answer}")
        print("=" * 60)
