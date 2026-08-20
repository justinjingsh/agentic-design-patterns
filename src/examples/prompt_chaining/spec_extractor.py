"""Prompt chaining example using AWS Bedrock and LangChain.

Demonstrates a two-step extraction pipeline:
1. Extract technical specifications from raw product text
2. Transform extracted specs into structured JSON format
"""

import json
import logging
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from ...app.bedrock import llm

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

    extraction_chain = extraction_prompt | llm | StrOutputParser()
    transformation_chain = transformation_prompt | llm | StrOutputParser()

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

    extracted_specs = extraction_chain.invoke({"text_input": raw_text})
    print()
    print("Extracted specifications:")
    print(extracted_specs)
    print()

    raw_json = transformation_chain.invoke({"specifications": extracted_specs})
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON output: %s", exc)
        print(f"[Warning] Chain output is not valid JSON, retry or manual check needed: {exc}")
        print("Raw output:", raw_json)
    else:
        print("Chain final output (structured JSON, passed validity check):")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        logger.info("JSON validation successful")
    print("-" * 60)


def extract_specifications() -> None:
    """Main entry point: build the extraction chain and process all sample product texts."""
    logger.info("Building specification extraction chain")
    print('Building specification extraction chain')
    extraction_chain, transformation_chain = build_chain()
    for text in RAW_TEXTS:
        run_and_validate(extraction_chain, transformation_chain, text)
    