"""CLI interface for running examples."""

import sys
import logging

from ..app.config import validate_aws_credentials
from ..app.cli import CMD_MULTIAGENT, CMD_PLANNING, CMD_REFLECTION, CMD_ROUTING, CMD_TOOLS, print_help, CMD_PROMPT_CHAINING, CMD_HELP, CMD_PARALLELIZATION

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def _run_prompt_chaining() -> None:
    """Run the prompt chaining example."""
    logger.info("Running prompt_chaining example")
    from .prompt_chaining import handle_requests
    handle_requests()


def _run_routing() -> None:
    """Run the routing example."""
    logger.info("Running routing example")
    from .routing import handle_requests
    handle_requests()

def _run_parallelization() -> None:
    """Run the parallelization example."""
    logger.info("Running parallelization example")
    from .parallelization import handle_requests
    handle_requests()

def _run_reflection() -> None:
    """Run the reflection example."""
    logger.info("Running reflection example")
    from .reflection import handle_requests
    handle_requests()

def _run_tools() -> None:
    """Run the tools example."""
    logger.info("Running tools example")
    from .tools import handle_requests
    handle_requests()

def _run_planning() -> None:
    """Run the planning example."""
    logger.info("Running planning example")
    from .planning import handle_requests
    handle_requests()

def _run_multiagent() -> None:
    """Run the multi-agent example."""
    logger.info("Running multi-agent example")
    from .multiagent import handle_requests
    handle_requests()

COMMANDS = {
    CMD_PROMPT_CHAINING: _run_prompt_chaining,
    CMD_ROUTING: _run_routing,
    CMD_PARALLELIZATION: _run_parallelization,
    CMD_REFLECTION: _run_reflection,
    CMD_TOOLS: _run_tools,
    CMD_PLANNING: _run_planning,
    CMD_MULTIAGENT: _run_multiagent,
    CMD_HELP: print_help,
}


def main() -> None:
    """Main CLI entry point."""
    _setup_logging()

    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1]
    logger.debug("Command invoked: %s", command)

    try:
        validate_aws_credentials()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        print(f"Configuration error: {e}")
        sys.exit(1)

    if command in COMMANDS:
        COMMANDS[command]()
    else:
        logger.error("Unknown example: %s", command)
        print(f"Unknown example: {command}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
