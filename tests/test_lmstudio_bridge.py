"""Unit tests for LM Studio TypeScript SDK bridge functionality.

This test suite defines the contract for the LM Studio SDK bridge that queries
model context length via Node.js subprocess. These tests verify:

1. Successful SDK query returns context length
2. Graceful fallback when Node.js not installed (FileNotFoundError)
3. Graceful fallback when SDK not installed (npm error)
4. Timeout handling (subprocess.TimeoutExpired)
5. Invalid JSON response handling

All tests are written in Red Phase - they should FAIL initially because the
`_query_lmstudio_context_limit` function does not exist yet.

The function contract:
- Inputs: model_name (str), base_url (str, WebSocket format "ws://localhost:1234")
- Outputs: context_length (int) or None on error
- Requirements:
  1. Call Node.js with inline JavaScript using @lmstudio/sdk
  2. 10-second timeout (accounts for Node.js startup)
  3. Graceful fallback on any error (returns None, doesn't raise)
  4. Parse JSON response with contextLength field
  5. Log errors at debug level (not warning/error)
"""

import subprocess
from unittest.mock import Mock

import pytest

# Try to import the function - if it doesn't exist, tests will fail as expected
try:
    from leann.embedding_compute import _query_lmstudio_context_limit
except ImportError:
    # Function doesn't exist yet (Red Phase) - create a placeholder that will fail
    def _query_lmstudio_context_limit(*args, **kwargs):
        raise NotImplementedError(
            "_query_lmstudio_context_limit not implemented yet - this is the Red Phase"
        )


class TestLMStudioBridge:
    """Tests for LM Studio TypeScript SDK bridge integration."""

    def test_query_lmstudio_success(self, monkeypatch):
        """Verify successful SDK query returns context length.

        When the Node.js subprocess successfully queries the LM Studio SDK,
        it should return a JSON response with contextLength field. The function
        should parse this and return the integer context length.
        """

        def mock_run(*args, **kwargs):
            # Verify timeout is set to 10 seconds
            assert kwargs.get("timeout") == 10, "Should use 10-second timeout for Node.js startup"

            # Verify capture_output and text=True are set
            assert kwargs.get("capture_output") is True, "Should capture stdout/stderr"
            assert kwargs.get("text") is True, "Should decode output as text"

            # Return successful JSON response
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"contextLength": 8192, "identifier": "custom-model"}'
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        # Test with typical LM Studio model
        limit = _query_lmstudio_context_limit(
            model_name="custom-model", base_url="ws://localhost:1234"
        )

        assert limit == 8192, "Should return context length from SDK response"

    def test_query_lmstudio_nodejs_not_found(self, monkeypatch):
        """Verify graceful fallback when Node.js not installed.

        When Node.js is not installed, subprocess.run will raise FileNotFoundError.
        The function should catch this and return None (graceful fallback to registry).
        """

        def mock_run(*args, **kwargs):
            raise FileNotFoundError("node: command not found")

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="custom-model", base_url="ws://localhost:1234"
        )

        assert limit is None, "Should return None when Node.js not installed"

    def test_query_lmstudio_sdk_not_installed(self, monkeypatch):
        """Verify graceful fallback when @lmstudio/sdk not installed.

        When the SDK npm package is not installed, Node.js will return non-zero
        exit code with error message in stderr. The function should detect this
        and return None.
        """

        def mock_run(*args, **kwargs):
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = (
                "Error: Cannot find module '@lmstudio/sdk'\nRequire stack:\n- /path/to/script.js"
            )
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="custom-model", base_url="ws://localhost:1234"
        )

        assert limit is None, "Should return None when SDK not installed"

    def test_query_lmstudio_timeout(self, monkeypatch):
        """Verify graceful fallback when subprocess times out.

        When the Node.js process takes longer than 10 seconds (e.g., LM Studio
        not responding), subprocess.TimeoutExpired should be raised. The function
        should catch this and return None.
        """

        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=["node", "lmstudio_bridge.js"], timeout=10)

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="custom-model", base_url="ws://localhost:1234"
        )

        assert limit is None, "Should return None on timeout"

    def test_query_lmstudio_invalid_json(self, monkeypatch):
        """Verify graceful fallback when response is invalid JSON.

        When the subprocess returns malformed JSON (e.g., due to SDK error),
        json.loads will raise ValueError/JSONDecodeError. The function should
        catch this and return None.
        """

        def mock_run(*args, **kwargs):
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "This is not valid JSON"
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="custom-model", base_url="ws://localhost:1234"
        )

        assert limit is None, "Should return None when JSON parsing fails"

    def test_query_lmstudio_missing_context_length_field(self, monkeypatch):
        """Verify graceful fallback when JSON lacks contextLength field.

        When the SDK returns valid JSON but without the expected contextLength
        field (e.g., error response), the function should return None.
        """

        def mock_run(*args, **kwargs):
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"identifier": "test-model", "error": "Model not found"}'
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="nonexistent-model", base_url="ws://localhost:1234"
        )

        assert limit is None, "Should return None when contextLength field missing"

    def test_query_lmstudio_null_context_length(self, monkeypatch):
        """Verify graceful fallback when contextLength is null.

        When the SDK returns contextLength: null (model couldn't be loaded),
        the function should return None for registry fallback.
        """

        def mock_run(*args, **kwargs):
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"contextLength": null, "identifier": "test-model"}'
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="test-model", base_url="ws://localhost:1234"
        )

        assert limit is None, "Should return None when contextLength is null"

    def test_query_lmstudio_zero_context_length(self, monkeypatch):
        """Verify graceful fallback when contextLength is zero.

        When the SDK returns contextLength: 0 (invalid value), the function
        should return None to trigger registry fallback.
        """

        def mock_run(*args, **kwargs):
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"contextLength": 0, "identifier": "test-model"}'
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="test-model", base_url="ws://localhost:1234"
        )

        assert limit is None, "Should return None when contextLength is zero"

    def test_query_lmstudio_with_custom_port(self, monkeypatch):
        """Verify SDK query works with non-default WebSocket port.

        LM Studio can run on custom ports. The function should pass the
        provided base_url to the Node.js subprocess.
        """

        def mock_run(*args, **kwargs):
            # Verify the base_url argument is passed correctly
            command = args[0] if args else kwargs.get("args", [])
            assert "ws://localhost:8080" in " ".join(command), (
                "Should pass custom port to subprocess"
            )

            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"contextLength": 4096, "identifier": "custom-model"}'
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="custom-model", base_url="ws://localhost:8080"
        )

        assert limit == 4096, "Should work with custom WebSocket port"

    @pytest.mark.parametrize(
        "context_length,expected",
        [
            (512, 512),  # Small context
            (2048, 2048),  # Common context
            (8192, 8192),  # Large context
            (32768, 32768),  # Very large context
        ],
    )
    def test_query_lmstudio_various_context_lengths(self, monkeypatch, context_length, expected):
        """Verify SDK query handles various context length values.

        Different models have different context lengths. The function should
        correctly parse and return any positive integer value.
        """

        def mock_run(*args, **kwargs):
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = f'{{"contextLength": {context_length}, "identifier": "test"}}'
            mock_result.stderr = ""
            return mock_result

        monkeypatch.setattr("subprocess.run", mock_run)

        limit = _query_lmstudio_context_limit(
            model_name="test-model", base_url="ws://localhost:1234"
        )

        assert limit == expected, f"Should return {expected} for context length {context_length}"

    def test_query_lmstudio_logs_at_debug_level(self, monkeypatch, caplog):
        """Verify errors are logged at DEBUG level, not WARNING/ERROR.

        Following the graceful fallback pattern from Ollama implementation,
        errors should be logged at debug level to avoid alarming users when
        fallback to registry works fine.
        """
        import logging

        caplog.set_level(logging.DEBUG, logger="leann.embedding_compute")

        def mock_run(*args, **kwargs):
            raise FileNotFoundError("node: command not found")

        monkeypatch.setattr("subprocess.run", mock_run)

        _query_lmstudio_context_limit(model_name="test-model", base_url="ws://localhost:1234")

        # Check that debug logging occurred (not warning/error)
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) > 0, "Should log error at DEBUG level"

        # Verify no WARNING or ERROR logs
        warning_or_error_logs = [
            record for record in caplog.records if record.levelname in ["WARNING", "ERROR"]
        ]
        assert len(warning_or_error_logs) == 0, (
            "Should not log at WARNING/ERROR level for expected failures"
        )
