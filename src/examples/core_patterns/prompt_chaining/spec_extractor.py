"""Prompt chaining example using AWS Bedrock and LangChain.

Demonstrates the Prompt Chaining agentic pattern: a task is split into a
fixed sequence of LLM calls where each call's output feeds the next. Here
it is a two-step pipeline:

  1. Extraction:     raw product text ── LLM ──> a one-sentence spec summary
  2. Transformation: that spec summary ── LLM ──> a strict JSON object

Neither step decides what comes next — the programmer wires the order in
advance, so every run does the same two calls in the same order. The only
runtime branching is error handling: step 2's output is parsed with
``json.loads()`` and a ``JSONDecodeError`` is reported rather than trusted,
because "just ask the model for JSON" is not reliable enough for production.

How this differs from the other patterns in this repo:
  - routing: the model makes one classification decision that changes which
    handler runs; here the steps are fixed regardless of the input.
  - parallelization: independent sub-tasks run concurrently; here step 2
    consumes step 1's output, so the calls must stay sequential.
  - reflection / planning / tools / multiagent: the number or the choice of
    steps is decided at runtime by the model; here both are hard-coded.

Flow for one product text:

    raw_text ── extraction_chain ──> "CPU ..., RAM ..., storage ..."
             ── transformation_chain ──> '{"cpu": ..., "memory": ..., "storage": ...}'
             ── json.loads() ──> validated dict   (or a logged JSONDecodeError)
"""

import json
import logging
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# Sample product descriptions for testing
RAW_TEXTS: list[str] = [
    "Brand new laptop equipped with 5th Gen Intel Core i9 processor (3.1GHz, 8-core), "
    "32GB DDR5 memory, 1TB NVMe SSD storage.",
    "Entry-level ultrabook with AMD Ryzen 5 processor (2.3GHz, 6-core), "
    "16GB LPDDR5 memory, 512GB PCIe SSD.",
]


def build_chain() -> tuple[Runnable[dict[str, Any], str], Runnable[dict[str, Any], str]]:
    """Build a two-step prompt chain for extracting and structuring product specs.

    Returns:
        A tuple of (extraction_chain, transformation_chain) where:
        - extraction_chain: raw text -> extracted specs (string)
        - transformation_chain: extracted specs -> structured JSON
    """
    # Step 1: Extract key technical specifications from raw text
    extraction_prompt = ChatPromptTemplate.from_template(
        "Extract technical specifications from the following text, listing key parameters in one concise sentence:\n\n{text_input}"
    )

    # Step 2: Convert extracted specs into structured JSON format
    transformation_prompt = ChatPromptTemplate.from_template(
        "Convert the following technical specifications into a JSON object with fixed field names: cpu, memory, storage. "
        "Output only the JSON itself, no explanations or markdown code blocks:\n\n{specifications}"
    )

    # Each step is a self-contained LCEL chain: format the prompt -> call the
    # LLM -> pull the plain string out of the response message.
    extraction_chain = extraction_prompt | llm | StrOutputParser()
    transformation_chain = transformation_prompt | llm | StrOutputParser()

    # The two chains are returned separately rather than fused into one
    # `extraction_chain | transformation_chain` pipe. Keeping them apart lets
    # run_and_validate() invoke them one at a time and print the intermediate
    # extraction result — seeing that hand-off is the point of the example.
    logger.debug("Prompt chaining pipelines built successfully")
    return extraction_chain, transformation_chain


def run_and_validate(
    extraction_chain: Runnable[dict[str, Any], str],
    transformation_chain: Runnable[dict[str, Any], str],
    raw_text: str,
) -> None:
    """Run the extraction chain and validate JSON output.

    This demonstrates best practices for structured output: even when requesting JSON,
    validation is required since the model may not always produce valid JSON.
    """
    print("Raw text:", raw_text)

    # Step 1. The raw text goes in under the key the extraction prompt expects
    # ({text_input}); the output is a free-form spec sentence, not yet JSON.
    extracted_specs = extraction_chain.invoke({"text_input": raw_text})
    print()
    print("Extracted specifications:")
    print(extracted_specs)
    print()

    # Step 2. Step 1's string is passed straight into step 2 as {specifications}
    # — this is the chain "link". The model is asked for bare JSON, but its
    # answer is still just text at this point.
    raw_json = transformation_chain.invoke({"specifications": extracted_specs})
    # Don't trust the model to have produced valid JSON — parse it and treat a
    # decode failure as a normal, expected outcome. The try/except/else split
    # keeps the happy path (else: runs only when no exception was raised)
    # cleanly separated from the failure path.
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON output: %s", exc)
        print(f"[Warning] Chain output is not valid JSON, retry or manual check needed: {exc}")
        print("Raw output:", raw_json)
    else:
        # Round-trip through json.dumps() to pretty-print and to prove the
        # parse produced a real Python object.
        print("Chain final output (structured JSON, passed validity check):")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        logger.info("JSON validation successful")
    print("-" * 60)


def handle_requests() -> None:
    """Main entry point: build the extraction chain and process all sample product texts."""
    logger.info("Building specification extraction chain")
    print('Building specification extraction chain')
    extraction_chain, transformation_chain = build_chain()
    for text in RAW_TEXTS:
        run_and_validate(extraction_chain, transformation_chain, text)
    