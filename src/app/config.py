import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

AWS_REGION: str = os.getenv('AWS_REGION', 'ap-southeast-2')
AWS_ACCESS_KEY_ID: str | None = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY: str | None = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN: str | None = os.getenv('AWS_SESSION_TOKEN')

BEDROCK_MODEL_ID: str = os.getenv('BEDROCK_MODEL_ID', 'amazon.nova-micro-v1:0')


def validate_aws_credentials() -> None:
	"""Validate that required AWS credentials are set."""
	if not AWS_ACCESS_KEY_ID:
		raise ValueError("AWS_ACCESS_KEY_ID not set in environment")
	if not AWS_SECRET_ACCESS_KEY:
		raise ValueError("AWS_SECRET_ACCESS_KEY not set in environment")
	logger.debug(f"AWS credentials validated (region: {AWS_REGION}, model: {BEDROCK_MODEL_ID})")