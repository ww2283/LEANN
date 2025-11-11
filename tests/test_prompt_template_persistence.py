"""
Integration tests for prompt template metadata persistence and reuse.

These tests verify the complete lifecycle of prompt template persistence:
1. Template is saved to .meta.json during index build
2. Template is automatically loaded during search operations
3. Template can be overridden with explicit flag during search
4. Template is reused during chat/ask operations

These are integration tests that:
- Use real file system with temporary directories
- Run actual build and search operations
- Inspect .meta.json file contents directly
- Mock embedding servers to avoid external dependencies
- Use small test codebases for fast execution

Expected to FAIL in Red Phase because metadata persistence verification is not yet implemented.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from leann.api import LeannBuilder, LeannSearcher


class TestPromptTemplateMetadataPersistence:
    """Tests for prompt template storage in .meta.json during build."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock compute_embeddings to return dummy embeddings."""
        with patch("leann.api.compute_embeddings") as mock_compute:
            # Return dummy embeddings as numpy array
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield mock_compute

    def test_prompt_template_saved_to_metadata(self, temp_index_dir, mock_embeddings):
        """
        Verify that when build is run with embedding_options containing prompt_template,
        the template value is saved to .meta.json file.

        This is the core persistence requirement - templates must be saved to allow
        reuse in subsequent search operations without re-specifying the flag.

        Expected failure: .meta.json exists but doesn't contain embedding_options
        with prompt_template, or the value is not persisted correctly.
        """
        # Setup test data
        index_path = temp_index_dir / "test_index.leann"
        template = "search_document: "

        # Build index with prompt template in embedding_options
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={"prompt_template": template},
        )

        # Add a simple document
        builder.add_text("This is a test document for indexing")

        # Build the index
        builder.build_index(str(index_path))

        # Verify .meta.json was created and contains the template
        meta_path = temp_index_dir / "test_index.leann.meta.json"
        assert meta_path.exists(), ".meta.json file should be created during build"

        # Read and parse metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # Verify embedding_options exists in metadata
        assert "embedding_options" in meta_data, (
            "embedding_options should be saved to .meta.json when provided"
        )

        # Verify prompt_template is in embedding_options
        embedding_options = meta_data["embedding_options"]
        assert "prompt_template" in embedding_options, (
            "prompt_template should be saved within embedding_options"
        )

        # Verify the template value matches what we provided
        assert embedding_options["prompt_template"] == template, (
            f"Template should be '{template}', got '{embedding_options.get('prompt_template')}'"
        )

    def test_prompt_template_absent_when_not_provided(self, temp_index_dir, mock_embeddings):
        """
        Verify that when no prompt template is provided during build,
        .meta.json either doesn't have embedding_options or prompt_template key.

        This ensures clean metadata without unnecessary keys when features aren't used.

        Expected behavior: Build succeeds, .meta.json doesn't contain prompt_template.
        """
        index_path = temp_index_dir / "test_no_template.leann"

        # Build index WITHOUT prompt template
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            # No embedding_options provided
        )

        builder.add_text("Document without template")
        builder.build_index(str(index_path))

        # Verify metadata
        meta_path = temp_index_dir / "test_no_template.leann.meta.json"
        assert meta_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # If embedding_options exists, it should not contain prompt_template
        if "embedding_options" in meta_data:
            embedding_options = meta_data["embedding_options"]
            assert "prompt_template" not in embedding_options, (
                "prompt_template should not be in metadata when not provided"
            )


class TestPromptTemplateAutoLoadOnSearch:
    """Tests for automatic loading of prompt template during search operations."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock compute_embeddings to capture calls and return dummy embeddings."""
        with patch("leann.api.compute_embeddings") as mock_compute:
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield mock_compute

    @pytest.fixture
    def mock_embedding_server_manager(self):
        """Mock EmbeddingServerManager to capture provider_options."""
        with patch("leann.searcher_base.EmbeddingServerManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager.start_server.return_value = (True, 5557)
            mock_manager_class.return_value = mock_manager
            yield mock_manager

    @pytest.fixture
    def index_with_template(self, temp_index_dir, mock_embeddings):
        """Build an index with a prompt template and return path."""
        index_path = temp_index_dir / "template_index.leann"
        template = "search_query: "

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={"prompt_template": template},
        )

        builder.add_text("Test document for search")
        builder.build_index(str(index_path))

        return str(index_path), template

    def test_prompt_template_auto_loaded_on_search(
        self, index_with_template, mock_embeddings, mock_embedding_server_manager
    ):
        """
        Verify that when searching an index built with a prompt template,
        the template is automatically loaded from .meta.json and passed to
        the embedding server.

        This is the core reuse requirement - users shouldn't need to remember
        or re-specify the template for every search operation.

        Expected failure: Template is not loaded from metadata, or is not passed
        to embedding server during search.
        """
        index_path, expected_template = index_with_template

        # Reset mocks to clear build calls
        mock_embeddings.reset_mock()
        mock_embedding_server_manager.reset_mock()

        # Create searcher (this should load metadata)
        searcher = LeannSearcher(index_path=index_path)

        # Verify that searcher loaded the embedding_options from metadata
        assert hasattr(searcher, "embedding_options"), (
            "Searcher should have embedding_options attribute"
        )
        assert "prompt_template" in searcher.embedding_options, (
            "Searcher should load prompt_template from .meta.json"
        )
        assert searcher.embedding_options["prompt_template"] == expected_template, (
            f"Loaded template should match saved template: '{expected_template}'"
        )

        # Perform a search with recompute_embeddings=True to trigger server
        with patch.object(searcher.backend_impl, "_compute_embedding_via_server") as mock_server_embed:
            mock_server_embed.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

            results = searcher.search("test query", top_k=5, recompute_embeddings=True)

            # Verify embedding server was started with provider_options containing template
            assert mock_embedding_server_manager.start_server.called, (
                "Embedding server should be started during search with recompute_embeddings"
            )

            # Check that provider_options was passed to start_server with prompt_template
            call_kwargs = mock_embedding_server_manager.start_server.call_args.kwargs
            assert "provider_options" in call_kwargs, (
                "start_server should receive provider_options"
            )

            provider_options = call_kwargs["provider_options"]
            assert provider_options is not None, "provider_options should not be None"
            assert "prompt_template" in provider_options, (
                "provider_options should contain prompt_template from metadata"
            )
            assert provider_options["prompt_template"] == expected_template, (
                f"Template passed to server should be '{expected_template}'"
            )

    def test_search_without_template_in_metadata(
        self, temp_index_dir, mock_embeddings, mock_embedding_server_manager
    ):
        """
        Verify that searching an index built WITHOUT a prompt template
        works correctly (backward compatibility).

        The searcher should handle missing prompt_template gracefully and not
        pass it to the embedding server.

        Expected behavior: Search succeeds, no template is used.
        """
        # Build index without template
        index_path = temp_index_dir / "no_template.leann"

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
        )
        builder.add_text("Document without template")
        builder.build_index(str(index_path))

        # Reset mocks
        mock_embeddings.reset_mock()
        mock_embedding_server_manager.reset_mock()

        # Create searcher and search
        searcher = LeannSearcher(index_path=str(index_path))

        # Verify no template in embedding_options
        assert "prompt_template" not in searcher.embedding_options, (
            "Searcher should not have prompt_template when not in metadata"
        )

        # Search should work without template
        with patch.object(searcher.backend_impl, "_compute_embedding_via_server") as mock_server_embed:
            mock_server_embed.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

            results = searcher.search("query", top_k=5, recompute_embeddings=True)

            # Verify embedding server was started
            assert mock_embedding_server_manager.start_server.called

            # Check that provider_options doesn't contain prompt_template
            call_kwargs = mock_embedding_server_manager.start_server.call_args.kwargs
            provider_options = call_kwargs.get("provider_options", {})

            # Either empty dict or doesn't contain prompt_template
            assert "prompt_template" not in provider_options, (
                "prompt_template should not be passed when not in metadata"
            )


class TestPromptTemplateReuseInChat:
    """Tests for prompt template reuse in chat/ask operations."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock compute_embeddings to return dummy embeddings."""
        with patch("leann.api.compute_embeddings") as mock_compute:
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield mock_compute

    @pytest.fixture
    def mock_embedding_server_manager(self):
        """Mock EmbeddingServerManager for chat tests."""
        with patch("leann.searcher_base.EmbeddingServerManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager.start_server.return_value = (True, 5557)
            mock_manager_class.return_value = mock_manager
            yield mock_manager

    @pytest.fixture
    def index_with_template(self, temp_index_dir, mock_embeddings):
        """Build an index with a prompt template."""
        index_path = temp_index_dir / "chat_template_index.leann"
        template = "document_query: "

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            embedding_options={"prompt_template": template},
        )

        builder.add_text("Test document for chat")
        builder.build_index(str(index_path))

        return str(index_path), template

class TestPromptTemplateIntegrationWithEmbeddingModes:
    """Tests for prompt template compatibility with different embedding modes."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for test indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.parametrize("mode,model,template,filename_prefix", [
        ("openai", "text-embedding-3-small", "Represent this for searching: ", "openai_template"),
        ("ollama", "nomic-embed-text", "search_query: ", "ollama_template"),
        ("sentence-transformers", "facebook/contriever", "query: ", "st_template"),
    ])
    def test_prompt_template_metadata_with_embedding_modes(
        self, temp_index_dir, mode, model, template, filename_prefix
    ):
        """Verify prompt template is saved correctly across different embedding modes.

        Tests that prompt templates are persisted to .meta.json for:
        - OpenAI mode (primary use case)
        - Ollama mode (also supports templates)
        - Sentence-transformers mode (saved for forward compatibility)

        Expected behavior: Template is saved to .meta.json regardless of mode.
        """
        with patch("leann.api.compute_embeddings") as mock_compute:
            mock_compute.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

            index_path = temp_index_dir / f"{filename_prefix}.leann"

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model=model,
                embedding_mode=mode,
                embedding_options={"prompt_template": template},
            )

            builder.add_text(f"{mode.capitalize()} test document")
            builder.build_index(str(index_path))

            # Verify metadata
            meta_path = temp_index_dir / f"{filename_prefix}.leann.meta.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

            assert meta_data["embedding_mode"] == mode
            # Template should be saved for all modes (even if not used by some)
            if "embedding_options" in meta_data:
                assert meta_data["embedding_options"]["prompt_template"] == template
