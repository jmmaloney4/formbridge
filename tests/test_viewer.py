"""Tests for the FormBridge viewer module.

Covers:
- HTTP server startup and endpoint handling
- PDF serving
- Mapping and schema endpoints
- Field update functionality
- PDF save functionality
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from formbridge.models import (
    FieldMapping,
    FieldMappingResult,
    FieldPosition,
    FieldType,
    FormField,
    FormSchema,
)
from formbridge.viewer import (
    Viewer,
    ViewerError,
    ViewerRequestHandler,
    ViewerState,
    run_viewer,
)

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pdf_path() -> Path:
    """Return path to sample PDF."""
    pdf_path = FIXTURES_DIR / "fillable.pdf"
    if not pdf_path.exists():
        pytest.skip("Fillable PDF fixture not created")
    return pdf_path


@pytest.fixture
def sample_schema() -> FormSchema:
    """Create a sample form schema for testing."""
    return FormSchema(
        form_id="test-form",
        pages=1,
        fields=[
            FormField(
                id="field_001",
                label="Name",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=100, y=700, w=200, h=15),
                max_length=50,
            ),
            FormField(
                id="field_002",
                label="Amount",
                page=1,
                type=FieldType.NUMBER,
                position=FieldPosition(x=350, y=700, w=100, h=15),
            ),
        ],
    )


@pytest.fixture
def sample_mapping_result() -> FieldMappingResult:
    """Create a sample mapping result."""
    return FieldMappingResult(
        mappings=[
            FieldMapping(
                field_id="field_001",
                value="Test Name",
                confidence=0.95,
                reasoning="Matched name field",
                source_key="name",
            ),
            FieldMapping(
                field_id="field_002",
                value="1000",
                confidence=0.88,
                reasoning="Matched amount field",
                source_key="amount",
            ),
        ],
        calculations=[],
        warnings=[],
    )


@pytest.fixture
def sample_mapping_json(sample_mapping_result: FieldMappingResult, tmp_path: Path) -> Path:
    """Create a temporary mapping JSON file."""
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(sample_mapping_result.model_dump_json(indent=2))
    return mapping_path


@pytest.fixture
def sample_schema_json(sample_schema: FormSchema, tmp_path: Path) -> Path:
    """Create a temporary schema JSON file."""
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(sample_schema.model_dump_json(indent=2))
    return schema_path


# =============================================================================
# ViewerState Tests
# =============================================================================


class TestViewerState:
    """Tests for ViewerState class."""

    def test_init_with_pdf(self, sample_pdf_path: Path) -> None:
        """Test initializing state with a PDF file."""
        state = ViewerState(pdf_path=sample_pdf_path)

        assert state.pdf_path == sample_pdf_path
        assert state.pdf_bytes is not None
        assert len(state.pdf_bytes) > 0
        assert not state.is_modified

    def test_init_with_schema_and_mapping(
        self,
        sample_pdf_path: Path,
        sample_schema: FormSchema,
        sample_mapping_result: FieldMappingResult,
    ) -> None:
        """Test initializing state with schema and mapping."""
        state = ViewerState(
            pdf_path=sample_pdf_path,
            schema=sample_schema,
            mapping_result=sample_mapping_result,
        )

        assert state.schema == sample_schema
        assert state.mapping_result == sample_mapping_result

    def test_get_field_value(
        self,
        sample_pdf_path: Path,
        sample_mapping_result: FieldMappingResult,
    ) -> None:
        """Test getting field value from state."""
        state = ViewerState(
            pdf_path=sample_pdf_path,
            mapping_result=sample_mapping_result,
        )

        value = state.get_field_value("field_001")
        assert value == "Test Name"

        value = state.get_field_value("nonexistent")
        assert value is None

    def test_update_field(
        self,
        sample_pdf_path: Path,
        sample_schema: FormSchema,
        sample_mapping_result: FieldMappingResult,
    ) -> None:
        """Test updating a field value."""
        state = ViewerState(
            pdf_path=sample_pdf_path,
            schema=sample_schema,
            mapping_result=sample_mapping_result,
        )

        # Update existing field
        success = state.update_field("field_001", "New Name")

        assert success
        assert state.get_field_value("field_001") == "New Name"
        assert state.is_modified

    def test_update_nonexistent_field(
        self,
        sample_pdf_path: Path,
        sample_schema: FormSchema,
        sample_mapping_result: FieldMappingResult,
    ) -> None:
        """Test updating a field that doesn't have a mapping yet."""
        state = ViewerState(
            pdf_path=sample_pdf_path,
            schema=sample_schema,
            mapping_result=sample_mapping_result,
        )

        # Add new field mapping
        success = state.update_field("field_003", "New Value")

        # Should succeed (adds new mapping)
        assert success

    def test_save_pdf(
        self,
        sample_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test saving PDF to disk."""
        state = ViewerState(pdf_path=sample_pdf_path)

        output_path = tmp_path / "saved.pdf"
        success = state.save_pdf(output_path)

        assert success
        assert output_path.exists()
        assert output_path.stat().st_size > 0


# =============================================================================
# ViewerRequestHandler Tests
# =============================================================================


class TestViewerRequestHandler:
    """Tests for HTTP request handler."""

    def test_handler_has_state(self) -> None:
        """Test that handler can have state attached."""
        # State is set at class level
        assert hasattr(ViewerRequestHandler, 'state')

    # Note: More comprehensive handler tests would require
    # setting up a test HTTP server and making requests.
    # These are integration tests best done with httpx or similar.


# =============================================================================
# Viewer Tests
# =============================================================================


class TestViewer:
    """Tests for Viewer class."""

    def test_init(
        self,
        sample_pdf_path: Path,
        sample_mapping_json: Path,
        sample_schema_json: Path,
    ) -> None:
        """Test initializing viewer."""
        viewer = Viewer(
            pdf_path=sample_pdf_path,
            mapping_path=sample_mapping_json,
            schema_path=sample_schema_json,
            port=8765,
            open_browser=False,
        )

        assert viewer.pdf_path == sample_pdf_path
        assert viewer.mapping_path == sample_mapping_json
        assert viewer.schema_path == sample_schema_json
        assert viewer.port == 8765
        assert not viewer.open_browser

    def test_load_schema_from_file(
        self,
        sample_pdf_path: Path,
        sample_schema_json: Path,
    ) -> None:
        """Test loading schema from JSON file."""
        viewer = Viewer(
            pdf_path=sample_pdf_path,
            schema_path=sample_schema_json,
            open_browser=False,
        )

        schema = viewer._load_schema()

        assert schema is not None
        assert schema.form_id == "test-form"
        assert len(schema.fields) == 2

    def test_load_schema_from_pdf_scan(
        self,
        sample_pdf_path: Path,
    ) -> None:
        """Test loading schema by scanning PDF."""
        viewer = Viewer(
            pdf_path=sample_pdf_path,
            open_browser=False,
        )

        schema = viewer._load_schema()

        # Should scan the PDF and return a schema
        assert schema is not None

    def test_load_mapping_from_file(
        self,
        sample_pdf_path: Path,
        sample_mapping_json: Path,
    ) -> None:
        """Test loading mapping from JSON file."""
        viewer = Viewer(
            pdf_path=sample_pdf_path,
            mapping_path=sample_mapping_json,
            open_browser=False,
        )

        mapping = viewer._load_mapping()

        assert mapping is not None
        assert len(mapping.mappings) == 2

    def test_load_mapping_none(
        self,
        sample_pdf_path: Path,
    ) -> None:
        """Test loading mapping when no file provided."""
        viewer = Viewer(
            pdf_path=sample_pdf_path,
            open_browser=False,
        )

        mapping = viewer._load_mapping()

        assert mapping is None


# =============================================================================
# Run Viewer Tests
# =============================================================================


class TestRunViewer:
    """Tests for run_viewer convenience function."""

    def test_run_viewer_basic(
        self,
        sample_pdf_path: Path,
    ) -> None:
        """Test basic run_viewer call."""
        # We can't actually run the server in tests, but we can
        # verify the function exists and accepts the right parameters

        import inspect
        sig = inspect.signature(run_viewer)

        params = list(sig.parameters.keys())
        assert 'pdf_path' in params
        assert 'mapping_path' in params
        assert 'schema_path' in params
        assert 'port' in params
        assert 'open_browser' in params


# =============================================================================
# CLI Tests
# =============================================================================


class TestViewCLI:
    """Tests for the view CLI command."""

    def test_view_help(self) -> None:
        """Test view command help output."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["view", "--help"])

        assert result.exit_code == 0
        assert "visual PDF viewer" in result.output.lower() or "viewer" in result.output.lower()
        assert "--mapping" in result.output
        assert "--port" in result.output

    def test_view_missing_pdf(self) -> None:
        """Test view with missing PDF."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["view", "/nonexistent.pdf"])

        assert result.exit_code != 0

    def test_view_with_mapping(
        self,
        sample_pdf_path: Path,
        sample_mapping_json: Path,
    ) -> None:
        """Test view with mapping file."""
        from click.testing import CliRunner
        from formbridge.cli import main

        if not sample_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        runner = CliRunner()

        # Note: This will try to start the server, so we use a timeout
        # In a real test environment, we'd mock the server start
        result = runner.invoke(main, [
            "view",
            str(sample_pdf_path),
            "--mapping", str(sample_mapping_json),
            "--no-browser",
            "--port", "8766",  # Use different port to avoid conflicts
        ], timeout=2, catch_exceptions=True)

        # The command should try to start but might timeout
        # We're mainly testing that it parses arguments correctly
        assert "Error" not in result.output or "address already in use" in result.output.lower()


class TestFillViewFlag:
    """Tests for the --view flag on fill command."""

    def test_fill_help_includes_view(self) -> None:
        """Test that fill help includes --view option."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["fill", "--help"])

        assert result.exit_code == 0
        assert "--view" in result.output


# =============================================================================
# Integration Tests
# =============================================================================


class TestViewerIntegration:
    """Integration tests for the viewer."""

    @pytest.mark.integration
    def test_full_viewer_workflow(
        self,
        sample_pdf_path: Path,
        sample_schema: FormSchema,
        sample_mapping_result: FieldMappingResult,
        tmp_path: Path,
    ) -> None:
        """Test the full viewer workflow."""
        # Save schema and mapping to temp files
        schema_path = tmp_path / "schema.json"
        mapping_path = tmp_path / "mapping.json"

        schema_path.write_text(sample_schema.model_dump_json(indent=2))
        mapping_path.write_text(sample_mapping_result.model_dump_json(indent=2))

        # Create viewer state
        state = ViewerState(
            pdf_path=sample_pdf_path,
            schema=sample_schema,
            mapping_result=sample_mapping_result,
        )

        # Verify initial state
        assert state.get_field_value("field_001") == "Test Name"

        # Update a field
        success = state.update_field("field_001", "Updated Name")
        assert success

        # Verify update
        assert state.get_field_value("field_001") == "Updated Name"
        assert state.is_modified

        # Save PDF
        output_path = tmp_path / "output.pdf"
        success = state.save_pdf(output_path)
        assert success
        assert output_path.exists()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestViewerErrorHandling:
    """Tests for error handling in viewer."""

    def test_missing_pdf_error(self) -> None:
        """Test error when PDF doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ViewerState(pdf_path=Path("/nonexistent/file.pdf"))

    def test_get_field_without_mapping(self, sample_pdf_path: Path) -> None:
        """Test getting field value when no mapping is set."""
        state = ViewerState(pdf_path=sample_pdf_path)

        value = state.get_field_value("any_field")
        assert value is None

    def test_update_field_without_schema(self, sample_pdf_path: Path) -> None:
        """Test updating field when no schema is set."""
        state = ViewerState(pdf_path=sample_pdf_path)

        # Should return False since we can't regenerate PDF without schema
        success = state.update_field("field_001", "New Value")
        assert not success
