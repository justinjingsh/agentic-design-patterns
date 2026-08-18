"""Tests for CLI utilities."""

from src.app.cli import EXAMPLES, CMD_PROMPT_CHAINING, CMD_HELP


class TestCliRegistry:
	"""Test CLI command registry."""

	def test_examples_dict_has_prompt_chaining(self):
		"""Test that prompt_chaining example is registered."""
		assert CMD_PROMPT_CHAINING in EXAMPLES
		assert isinstance(EXAMPLES[CMD_PROMPT_CHAINING], str)
		assert len(EXAMPLES[CMD_PROMPT_CHAINING]) > 0

	def test_examples_all_have_descriptions(self):
		"""Test that all examples have descriptions."""
		for name, description in EXAMPLES.items():
			assert isinstance(name, str)
			assert isinstance(description, str)
			assert len(description) > 0

	def test_help_command_constant(self):
		"""Test that help command constant is defined."""
		assert CMD_HELP == "help"
		assert isinstance(CMD_HELP, str)
