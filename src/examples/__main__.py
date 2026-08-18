"""CLI interface for running examples."""

import sys
import logging
from ..app.config import validate_aws_credentials
from ..app.cli import print_help, CMD_PROMPT_CHAINING, CMD_HELP

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
	"""Configure logging for the application."""
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	)


def main() -> None:
	"""Main CLI entry point."""
	_setup_logging()

	if len(sys.argv) < 2:
		print_help()
		sys.exit(1)

	command = sys.argv[1]
	logger.debug(f"Command invoked: {command}")

	try:
		validate_aws_credentials()
	except ValueError as e:
		logger.error(f"Configuration error: {e}")
		print(f"Configuration error: {e}")
		sys.exit(1)

	if command == CMD_PROMPT_CHAINING:
		logger.info("Running prompt_chaining example")
		from .prompt_chaining import extract_specifications
		extract_specifications()
	elif command == CMD_HELP:
		print_help()
	else:
		logger.error(f"Unknown example: {command}")
		print(f"Unknown example: {command}")
		print_help()
		sys.exit(1)


if __name__ == "__main__":
	main()
