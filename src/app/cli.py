"""Shared CLI utilities for examples."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Command constants
CMD_PROMPT_CHAINING: str = "prompt_chaining"
CMD_ROUTING: str = "routing"
CMD_PARALLELIZATION: str = "parallelization"
CMD_REFLECTION: str = "reflection"
CMD_HELP: str = "help"

EXAMPLES: Dict[str, str] = {
    CMD_PROMPT_CHAINING: "Extract specs and transform to JSON",
    CMD_ROUTING: "Task Cordinator Agent",
    CMD_PARALLELIZATION: "Run tasks in parallel",
    CMD_REFLECTION: "Self-reflecting agent",
}


def print_help() -> None:
    """Print available examples."""
    help_text = "Agentic Design Patterns - Examples\n"
    help_text += "=" * 50 + "\n\n"
    help_text += "Usage: uv run python -m src.examples <example>\n\n"
    help_text += "Available examples:\n"
    for name, description in EXAMPLES.items():
        help_text += f"  {name:<20} - {description}\n"
    help_text += "  help                 - Show this help message\n\n"
    help_text += "Example:\n"
    help_text += "  uv run python -m src.examples prompt_chaining\n"
    help_text += "  uv run python -m src.examples routing\n"
    help_text += "  uv run python -m src.examples parallelization\n"
    help_text += "  uv run python -m src.examples reflection\n"

    print(help_text)
    logger.info("Help text displayed")
