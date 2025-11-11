"""Unit tests for prompt template prepending in OpenAI embeddings.

This test suite defines the contract for prompt template functionality that allows
users to prepend a consistent prompt to all embedding inputs. These tests verify:

1. Template prepending to all input texts before embedding computation
2. Graceful handling of None/missing provider_options
3. Empty string template behavior (no-op)
4. Logging of template application for observability
5. Template application before token truncation

All tests are written in Red Phase - they should FAIL initially because the
implementation does not exist yet.
"""

import logging
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from leann.embedding_compute import compute_embeddings_openai


class TestPromptTemplatePrepending:
    """Tests for prompt template prepending in compute_embeddings_openai."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client that captures input texts."""
        mock_client = MagicMock()

        # Mock the embeddings.create response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        return mock_client

    @pytest.fixture
    def mock_openai_module(self, mock_openai_client):
        """Mock the openai module to return our mock client."""
        # openai is imported inside the function, so we need to patch it there
        with patch("openai.OpenAI", return_value=mock_openai_client) as mock_openai:
            yield mock_openai

    def test_prompt_template_prepended_to_all_texts(self, mock_openai_module, mock_openai_client):
        """Verify template is prepended to all input texts.

        When provider_options contains "prompt_template", that template should
        be prepended to every text in the input list before sending to OpenAI API.

        This is the core functionality: the template acts as a consistent prefix
        that provides context or instruction for the embedding model.
        """
        texts = ["First document", "Second document"]
        template = "search_document: "
        provider_options = {"prompt_template": template}

        # Call compute_embeddings_openai with provider_options
        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify embeddings.create was called with templated texts
        mock_openai_client.embeddings.create.assert_called_once()
        call_args = mock_openai_client.embeddings.create.call_args

        # Extract the input texts sent to API
        sent_texts = call_args.kwargs["input"]

        # Verify template was prepended to all texts
        assert len(sent_texts) == 2, "Should send same number of texts"
        assert sent_texts[0] == "search_document: First document", (
            "Template should be prepended to first text"
        )
        assert sent_texts[1] == "search_document: Second document", (
            "Template should be prepended to second text"
        )

        # Verify result is valid embeddings array
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3), "Should return correct shape"

    def test_prompt_template_none_provider_options_is_noop(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify None provider_options doesn't modify texts.

        When provider_options is None (not provided), texts should be
        sent to the API unchanged. This is the default behavior.
        """
        texts = ["Original text one", "Original text two"]

        # Call without provider_options (None)
        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=None,
        )

        # Verify texts were sent unchanged
        mock_openai_client.embeddings.create.assert_called_once()
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]

        assert sent_texts[0] == "Original text one", "Text should be unchanged"
        assert sent_texts[1] == "Original text two", "Text should be unchanged"

        # Verify result is valid
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    def test_prompt_template_missing_key_is_noop(self, mock_openai_module, mock_openai_client):
        """Verify missing prompt_template key doesn't modify texts.

        When provider_options is provided but doesn't contain "prompt_template",
        texts should be sent unchanged. This allows provider_options to be used
        for other settings (base_url, api_key) without affecting template behavior.
        """
        texts = ["Text without template", "Another text"]
        provider_options = {"base_url": "https://api.openai.com/v1"}

        # Call with provider_options but no prompt_template key
        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify texts were sent unchanged
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]

        assert sent_texts[0] == "Text without template"
        assert sent_texts[1] == "Another text"

        # Verify result is valid
        assert isinstance(result, np.ndarray)

    def test_prompt_template_empty_string(self, mock_openai_module, mock_openai_client):
        """Verify empty string template works correctly.

        When template is an empty string "", it should prepend nothing
        (effectively a no-op). This allows users to explicitly disable
        templating by setting it to empty string.
        """
        texts = ["Text one", "Text two"]
        provider_options = {"prompt_template": ""}

        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify texts are unchanged (empty string prepended)
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]

        assert sent_texts[0] == "Text one", "Empty template should not modify text"
        assert sent_texts[1] == "Text two", "Empty template should not modify text"

        assert isinstance(result, np.ndarray)

    def test_prompt_template_logged_for_observability(
        self, mock_openai_module, mock_openai_client, caplog
    ):
        """Verify template application is logged for observability.

        When a prompt template is applied, it should be logged at INFO level
        to help with debugging and understanding what texts were sent to the API.
        This is important for troubleshooting embedding quality issues.
        """
        texts = ["Document to embed"]
        template = "query: "
        provider_options = {"prompt_template": template}

        with caplog.at_level(logging.INFO):
            compute_embeddings_openai(
                texts=texts,
                model_name="text-embedding-3-small",
                provider_options=provider_options,
            )

        # Verify log contains information about template application
        log_messages = [record.message for record in caplog.records]

        # Should log that template is being applied
        template_logs = [msg for msg in log_messages if "prompt template" in msg.lower()]
        assert len(template_logs) > 0, (
            "Should log template application for observability"
        )

        # Log should include the template text
        template_mentioned = any(template in msg for msg in log_messages)
        assert template_mentioned, "Log should mention the template text"

    def test_prompt_template_with_multiple_batches(self, mock_openai_module, mock_openai_client):
        """Verify template is prepended in all batches when texts exceed batch size.

        OpenAI API has batch size limits. When input texts are split into
        multiple batches, the template should be prepended to texts in every batch.

        This ensures consistency across all API calls.
        """
        # Create many texts that will be split into multiple batches
        texts = [f"Document {i}" for i in range(1000)]
        template = "passage: "
        provider_options = {"prompt_template": template}

        # Mock multiple batch responses
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(1000)]
        mock_openai_client.embeddings.create.return_value = mock_response

        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify embeddings.create was called multiple times (batching)
        assert mock_openai_client.embeddings.create.call_count >= 2, (
            "Should make multiple API calls for large text list"
        )

        # Verify template was prepended in ALL batches
        for call in mock_openai_client.embeddings.create.call_args_list:
            sent_texts = call.kwargs["input"]
            for text in sent_texts:
                assert text.startswith(template), (
                    f"All texts in all batches should start with template. Got: {text}"
                )

        # Verify result shape
        assert result.shape[0] == 1000, "Should return embeddings for all texts"

    def test_prompt_template_preserves_original_texts_list(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify original texts list is not modified by template prepending.

        Template prepending should create new strings, not modify the input list.
        This prevents unexpected side effects for the caller.
        """
        original_texts = ["Original text one", "Original text two"]
        texts_copy = original_texts.copy()
        template = "prefix: "
        provider_options = {"prompt_template": template}

        compute_embeddings_openai(
            texts=original_texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify original list is unchanged
        assert original_texts == texts_copy, (
            "Original texts list should not be modified"
        )
        assert original_texts[0] == "Original text one", (
            "First text should be unchanged"
        )
        assert original_texts[1] == "Original text two", (
            "Second text should be unchanged"
        )

    def test_prompt_template_with_special_characters(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify template with special characters is handled correctly.

        Templates may contain special characters, Unicode, newlines, etc.
        These should all be prepended correctly without encoding issues.
        """
        texts = ["Document content"]
        # Template with various special characters
        template = "🔍 Search query [EN]: "
        provider_options = {"prompt_template": template}

        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify special characters in template were preserved
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]

        assert sent_texts[0] == "🔍 Search query [EN]: Document content", (
            "Special characters in template should be preserved"
        )

        assert isinstance(result, np.ndarray)

    def test_prompt_template_integration_with_existing_validation(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify template works with existing input validation.

        compute_embeddings_openai has validation for empty texts and whitespace.
        Template prepending should happen AFTER validation, so validation errors
        are thrown based on original texts, not templated texts.

        This ensures users get clear error messages about their input.
        """
        # Empty text should still raise ValueError even with template
        texts = [""]
        provider_options = {"prompt_template": "prefix: "}

        with pytest.raises(ValueError, match="empty/invalid"):
            compute_embeddings_openai(
                texts=texts,
                model_name="text-embedding-3-small",
                provider_options=provider_options,
            )

    def test_prompt_template_with_api_key_and_base_url(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify template works alongside other provider_options.

        provider_options may contain multiple settings: prompt_template,
        base_url, api_key. All should work together correctly.
        """
        texts = ["Test document"]
        provider_options = {
            "prompt_template": "embed: ",
            "base_url": "https://custom.api.com/v1",
            "api_key": "test-key-123",
        }

        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify template was applied
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]
        assert sent_texts[0] == "embed: Test document"

        # Verify OpenAI client was created with correct base_url
        mock_openai_module.assert_called()
        client_init_kwargs = mock_openai_module.call_args.kwargs
        assert client_init_kwargs["base_url"] == "https://custom.api.com/v1"
        assert client_init_kwargs["api_key"] == "test-key-123"

        assert isinstance(result, np.ndarray)


