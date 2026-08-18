"""Tests for the prompt chaining specification extractor example."""

import json
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.runnables import Runnable
from src.examples.prompt_chaining.spec_extractor import RAW_TEXTS


class TestChainConstruction:
	"""Test that the prompt chaining pipelines are built correctly."""

	def test_build_chain_returns_two_runnables(self):
		"""Test that build_chain returns a tuple of two runnables."""
		from src.examples.prompt_chaining.spec_extractor import build_chain
		extraction_chain, transformation_chain = build_chain()
		assert isinstance(extraction_chain, Runnable)
		assert isinstance(transformation_chain, Runnable)

	def test_raw_texts_not_empty(self):
		"""Test that sample texts are defined."""
		assert len(RAW_TEXTS) > 0
		for text in RAW_TEXTS:
			assert isinstance(text, str)
			assert len(text) > 0


class TestRunAndValidate:
	"""Test the run_and_validate function with mocked chains."""

	def test_run_and_validate_with_valid_json(self, capsys):
		"""Test run_and_validate with valid JSON output."""
		from src.examples.prompt_chaining.spec_extractor import run_and_validate

		extraction_chain = MagicMock()
		transformation_chain = MagicMock()

		extraction_chain.invoke.return_value = "CPU: i9, Memory: 32GB, Storage: 1TB"
		transformation_chain.invoke.return_value = '{"cpu": "i9", "memory": "32GB", "storage": "1TB"}'

		run_and_validate(extraction_chain, transformation_chain, "test specs")

		captured = capsys.readouterr()
		# Check for key output indicators (case-insensitive)
		output_lower = captured.out.lower()
		assert "json" in output_lower
		assert "cpu" in output_lower

	def test_run_and_validate_with_invalid_json(self, capsys):
		"""Test run_and_validate handles invalid JSON gracefully."""
		from src.examples.prompt_chaining.spec_extractor import run_and_validate

		extraction_chain = MagicMock()
		transformation_chain = MagicMock()

		extraction_chain.invoke.return_value = "CPU: i9, Memory: 32GB, Storage: 1TB"
		transformation_chain.invoke.return_value = "not valid json at all"

		run_and_validate(extraction_chain, transformation_chain, "test specs")

		captured = capsys.readouterr()
		output_lower = captured.out.lower()
		assert "warning" in output_lower or "json" in output_lower

	def test_run_and_validate_invokes_chains(self):
		"""Test that run_and_validate invokes both chains with correct inputs."""
		from src.examples.prompt_chaining.spec_extractor import run_and_validate

		extraction_chain = MagicMock()
		transformation_chain = MagicMock()

		extraction_chain.invoke.return_value = "CPU: i9"
		transformation_chain.invoke.return_value = '{"cpu": "i9"}'

		run_and_validate(extraction_chain, transformation_chain, "test specs")

		extraction_chain.invoke.assert_called_once_with({"text_input": "test specs"})
		transformation_chain.invoke.assert_called_once()
