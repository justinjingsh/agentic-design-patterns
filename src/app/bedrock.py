"""Shared Bedrock LLM instance for all examples."""

import logging
from langchain_aws import ChatBedrock
from .config import AWS_REGION, BEDROCK_MODEL_ID, validate_aws_credentials

logger = logging.getLogger(__name__)

validate_aws_credentials()

llm: ChatBedrock = ChatBedrock(
	model_id=BEDROCK_MODEL_ID,
	region_name=AWS_REGION,
)

logger.debug("ChatBedrock instance initialized (model: %s, region: %s)", BEDROCK_MODEL_ID, AWS_REGION)

__all__ = ['llm']
