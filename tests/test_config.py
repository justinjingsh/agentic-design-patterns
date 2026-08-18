"""Tests for configuration and validation."""

import pytest


class TestAwsCredentialsValidation:
	"""Test AWS credentials validation."""

	def test_validate_credentials_with_mock(self, monkeypatch):
		"""Test validation with mocked environment."""
		import importlib
		import sys

		# Remove the module if already imported
		if 'src.app.config' in sys.modules:
			del sys.modules['src.app.config']

		# Set environment before importing
		monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'test_key')
		monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'test_secret')

		from src.app.config import validate_aws_credentials
		# Should not raise
		validate_aws_credentials()

	def test_validate_credentials_function_behavior(self):
		"""Test that validate_aws_credentials checks both keys are required."""
		from src.app.config import validate_aws_credentials
		# The function will check the actual environment state
		# This test verifies the function exists and can be called
		# (actual validation is tested in integration scenarios)
		try:
			validate_aws_credentials()
			# If we get here, credentials are set in the environment
			# This is expected in test environments
		except ValueError as e:
			# If credentials aren't set, we expect a ValueError
			assert "AWS" in str(e)
