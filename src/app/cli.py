"""Shared CLI utilities for examples."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Command constants
CMD_PROMPT_CHAINING: str = "prompt_chaining"
CMD_ROUTING: str = "routing"
CMD_PARALLELIZATION: str = "parallelization"
CMD_REFLECTION: str = "reflection"
CMD_TOOLS: str = "tools"
CMD_PLANNING: str = "planning"
CMD_MULTIAGENT: str = "multiagent"
CMD_STATE: str = "state"
CMD_GOAL_MONITORING: str = "goal_monitoring"
CMD_EXCEPTION_HANDLING: str = "exception_handling"
CMD_HITL: str = "hitl"
CMD_RAG: str = "rag"
CMD_A2A: str = "a2a"
CMD_HELP: str = "help"

EXAMPLES: Dict[str, str] = {
    CMD_PROMPT_CHAINING: "Extract specs and transform to JSON",
    CMD_ROUTING: "Task Cordinator Agent",
    CMD_PARALLELIZATION: "Run tasks in parallel",
    CMD_REFLECTION: "Self-reflecting agent",
    CMD_TOOLS: "Tool usage agent",
    CMD_PLANNING: "Planning agent",
    CMD_MULTIAGENT: "Multi-agent collaboration",
    CMD_STATE: "State management across turns (short-term, summary, long-term memory)",
    CMD_GOAL_MONITORING: "Goal setting and monitoring against measurable criteria",
    CMD_EXCEPTION_HANDLING: "Recover from tool failures (retry, fallback, graceful degradation)",
    CMD_HITL: "Human-in-the-loop approval gate on side-effecting tool calls",
    CMD_RAG: "Retrieval-augmented generation: ground answers in a document corpus, cite sources, decline when unsupported",
    CMD_A2A: "Agent-to-Agent protocol: discover remote agents by card, delegate tasks over a transport with a lifecycle",
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
    help_text += "  uv run python -m src.examples tools\n"
    help_text += "  uv run python -m src.examples planning\n"
    help_text += "  uv run python -m src.examples multiagent\n"
    help_text += "  uv run python -m src.examples state\n"
    help_text += "  uv run python -m src.examples goal_monitoring\n"
    help_text += "  uv run python -m src.examples exception_handling\n"
    help_text += "  uv run python -m src.examples hitl\n"
    help_text += "  uv run python -m src.examples rag\n"
    help_text += "  uv run python -m src.examples a2a\n"

    print(help_text)
    logger.info("Help text displayed")
