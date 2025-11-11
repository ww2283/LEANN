"""
Tests for CLI argument integration of --embedding-prompt-template.

These tests verify that:
1. The --embedding-prompt-template flag is properly registered on build and search commands
2. The template value flows from CLI args to embedding_options dict
3. The template is passed through to compute_embeddings() function
4. Default behavior (no flag) is handled correctly
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from leann.cli import LeannCLI


@pytest.mark.parametrize("command,base_args,expected_command,template_value", [
    ("build", ["test-index", "--docs", "/tmp/test-docs"], "build", "search_query: "),
    ("search", ["test-index", "my query"], "search", "search_query: "),
])
class TestCLIPromptTemplateArgument:
    """Tests for --embedding-prompt-template on build and search commands."""

    def test_command_accepts_prompt_template_argument(
        self, command, base_args, expected_command, template_value
    ):
        """Verify that command parser accepts --embedding-prompt-template flag."""
        cli = LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(
            [command] + base_args + ["--embedding-prompt-template", template_value]
        )

        assert args.command == expected_command
        assert hasattr(args, "embedding_prompt_template"), \
            f"{command} command should have embedding_prompt_template attribute"
        assert args.embedding_prompt_template == template_value

    def test_command_prompt_template_default_is_none(
        self, command, base_args, expected_command, template_value
    ):
        """Verify default value is None when flag not provided (backward compatibility)."""
        cli = LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args([command] + base_args)

        assert hasattr(args, "embedding_prompt_template"), \
            f"{command} command should have embedding_prompt_template attribute"
        assert args.embedding_prompt_template is None, \
            "Default value should be None when flag not provided"


class TestBuildCommandPromptTemplateArgumentExtras:
    """Additional build-specific tests for prompt template argument."""

    def test_build_command_prompt_template_with_multiword_value(self):
        """
        Verify that template values with spaces are handled correctly.

        Templates like "search_document: " or "Represent this sentence for searching: "
        should be accepted as a single string argument.
        """
        cli = LeannCLI()
        parser = cli.create_parser()

        template = "Represent this sentence for searching: "
        args = parser.parse_args([
            "build",
            "test-index",
            "--docs", "/tmp/test-docs",
            "--embedding-prompt-template", template
        ])

        assert args.embedding_prompt_template == template


class TestPromptTemplateStoredInEmbeddingOptions:
    """Tests for template storage in embedding_options dict."""

    @patch("leann.cli.LeannBuilder")
    def test_prompt_template_stored_in_embedding_options_on_build(
        self, mock_builder_class, tmp_path
    ):
        """
        Verify that when --embedding-prompt-template is provided to build command,
        the value is stored in embedding_options dict passed to LeannBuilder.

        This test will fail because the CLI doesn't currently process this argument
        and add it to embedding_options.
        """
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        # Create CLI and run build command
        cli = LeannCLI()

        # Mock load_documents to return a document so builder is created
        cli.load_documents = Mock(return_value=[
            {"text": "test content", "metadata": {}}
        ])

        parser = cli.create_parser()

        template = "search_query: "
        args = parser.parse_args([
            "build",
            "test-index",
            "--docs", str(tmp_path),
            "--embedding-prompt-template", template,
            "--force"  # Force rebuild to ensure LeannBuilder is called
        ])

        # Run the build command
        import asyncio
        asyncio.run(cli.build_index(args))

        # Check that LeannBuilder was called with embedding_options containing prompt_template
        call_kwargs = mock_builder_class.call_args.kwargs
        assert "embedding_options" in call_kwargs, \
            "LeannBuilder should receive embedding_options"

        embedding_options = call_kwargs["embedding_options"]
        assert embedding_options is not None, \
            "embedding_options should not be None when template provided"
        assert "prompt_template" in embedding_options, \
            "embedding_options should contain 'prompt_template' key"
        assert embedding_options["prompt_template"] == template, \
            f"Template should be '{template}', got {embedding_options.get('prompt_template')}"

    @patch("leann.cli.LeannBuilder")
    def test_prompt_template_not_in_options_when_not_provided(
        self, mock_builder_class, tmp_path
    ):
        """
        Verify that when --embedding-prompt-template is NOT provided,
        embedding_options either doesn't have the key or it's None.

        This ensures we don't pass empty/None values unnecessarily.
        """
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        cli = LeannCLI()

        # Mock load_documents to return a document so builder is created
        cli.load_documents = Mock(return_value=[
            {"text": "test content", "metadata": {}}
        ])

        parser = cli.create_parser()

        args = parser.parse_args([
            "build",
            "test-index",
            "--docs", str(tmp_path),
            "--force"  # Force rebuild to ensure LeannBuilder is called
        ])

        import asyncio
        asyncio.run(cli.build_index(args))

        # Check that if embedding_options is passed, it doesn't have prompt_template
        call_kwargs = mock_builder_class.call_args.kwargs
        if "embedding_options" in call_kwargs and call_kwargs["embedding_options"]:
            embedding_options = call_kwargs["embedding_options"]
            # Either the key shouldn't exist, or it should be None
            assert "prompt_template" not in embedding_options or \
                   embedding_options["prompt_template"] is None, \
                   "prompt_template should not be set when flag not provided"


class TestPromptTemplateFlowsToComputeEmbeddings:
    """Tests for template flowing through to compute_embeddings function."""

    @patch("leann.api.compute_embeddings")
    def test_prompt_template_flows_to_compute_embeddings_via_provider_options(
        self, mock_compute_embeddings, tmp_path
    ):
        """
        Verify that the prompt template flows from CLI args through LeannBuilder
        to compute_embeddings() function via provider_options parameter.

        This is an integration test that verifies the complete flow:
        CLI → embedding_options → LeannBuilder → compute_embeddings(provider_options)

        This test will fail because:
        1. CLI doesn't capture the argument yet
        2. embedding_options doesn't include prompt_template
        3. LeannBuilder doesn't pass it through to compute_embeddings
        """
        # Mock compute_embeddings to return dummy embeddings as numpy array
        import numpy as np
        mock_compute_embeddings.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        # Use real LeannBuilder (not mocked) to test the actual flow
        cli = LeannCLI()

        # Mock load_documents to return a simple document
        cli.load_documents = Mock(return_value=[
            {"text": "test content", "metadata": {}}
        ])

        parser = cli.create_parser()

        template = "search_document: "
        args = parser.parse_args([
            "build",
            "test-index",
            "--docs", str(tmp_path),
            "--embedding-prompt-template", template,
            "--backend-name", "hnsw",  # Use hnsw backend
            "--force"  # Force rebuild to ensure index is created
        ])

        # This should fail because the flow isn't implemented yet
        import asyncio
        asyncio.run(cli.build_index(args))

        # Verify compute_embeddings was called with provider_options containing prompt_template
        assert mock_compute_embeddings.called, \
            "compute_embeddings should have been called"

        # Check the call arguments
        call_kwargs = mock_compute_embeddings.call_args.kwargs
        assert "provider_options" in call_kwargs, \
            "compute_embeddings should receive provider_options parameter"

        provider_options = call_kwargs["provider_options"]
        assert provider_options is not None, \
            "provider_options should not be None"
        assert "prompt_template" in provider_options, \
            "provider_options should contain prompt_template key"
        assert provider_options["prompt_template"] == template, \
            f"Template should be '{template}', got {provider_options.get('prompt_template')}"


class TestPromptTemplateArgumentHelp:
    """Tests for argument help text and documentation."""

    def test_build_command_prompt_template_has_help_text(self):
        """
        Verify that --embedding-prompt-template has descriptive help text.

        Good help text is crucial for CLI usability.
        """
        cli = LeannCLI()
        parser = cli.create_parser()

        # Get the build subparser
        # This is a bit tricky - we need to parse to get the help
        # We'll check that the help includes relevant keywords
        import io
        import sys
        from contextlib import redirect_stdout

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                parser.parse_args(["build", "--help"])
        except SystemExit:
            pass  # --help causes sys.exit(0)

        help_text = f.getvalue()
        assert "--embedding-prompt-template" in help_text, \
            "Help text should mention --embedding-prompt-template"
        # Check for keywords that should be in the help
        help_lower = help_text.lower()
        assert any(keyword in help_lower for keyword in ["template", "prompt", "prepend"]), \
            "Help text should explain what the prompt template does"

    def test_search_command_prompt_template_has_help_text(self):
        """
        Verify that search command also has help text for --embedding-prompt-template.
        """
        cli = LeannCLI()
        parser = cli.create_parser()

        import io
        import sys
        from contextlib import redirect_stdout

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                parser.parse_args(["search", "--help"])
        except SystemExit:
            pass  # --help causes sys.exit(0)

        help_text = f.getvalue()
        assert "--embedding-prompt-template" in help_text, \
            "Search help text should mention --embedding-prompt-template"
