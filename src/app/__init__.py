"""Main application module."""

import logging
from .config import validate_aws_credentials
from .cli import print_help

logger = logging.getLogger(__name__)


def main() -> None:
	"""Main application entry point."""
	try:
		validate_aws_credentials()
		logger.info("AWS credentials validated successfully")
		print_help()
	except ValueError as e:
		logger.error("Configuration error: %s", e)
		raise