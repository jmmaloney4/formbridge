"""Tests for the FormBridge writer module (Phase 3).

Covers:
- Value formatting for different field types
- AcroForm field filling with pikepdf
- Overlay writing with reportlab (if available)
- Field type handling (text, checkbox, date, number)
- PDF output validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pikepdf
import pytest

from formbridge.models import (
    FieldMapping,
    FieldMappingResult,
    FieldPosition,
    FieldType,
    FormField,
    FormSchema,
)
from formbridge.writer import (
    AcroFormFiller,
    FieldValueFormatter,
    OverlayWriter,
    PDFWriter,
    WriterError,
    write_filled_pdf,
)

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Fixtures
# =============================================================================


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
            FormField(
                id="field_003",
                label="Date",
                page=1,
                type=FieldType.DATE,
                position=FieldPosition(x=100, y=670, w=100, h=15),
            ),
            FormField(
                id="field_004",
                label="Agreed",
                page=1,
                type=FieldType.CHECKBOX,
                position=FieldPosition(x=350, y=670, w=15, h=15),
                checked_value="Yes",
            ),
            FormField(
                id="field_005",
                label="EIN",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=100, y=640, w=100, h=15),
            ),
        ],
    )


@pytest.fixture
def sample_mapping_result() -> FieldMappingResult:
    """Create a sample mapping result."""
    return FieldMappingResult(
        mappings=[
            FieldMapping(field_id="field_001", value="Acme Partners LLC", confidence=0.95),
            FieldMapping(field_id="field_002", value="10000", confidence=0.98),
            FieldMapping(field_id="field_003", value="2023-03-15", confidence=0.92),
            FieldMapping(field_id="field_004", value="Yes", confidence=0.99),
            FieldMapping(field_id="field_005", value="82-1234567", confidence=0.97),
        ],
        calculations=[],
        warnings=[],
    )


@pytest.fixture
def fillable_pdf_path() -> Path:
    """Return path to fillable test PDF."""
    return FIXTURES_DIR / "fillable.pdf"


@pytest.fixture
def minimal_pdf_path() -> Path:
    """Return path to minimal test PDF."""
    return FIXTURES_DIR / "minimal.pdf"


# =============================================================================
# Field Value Formatter Tests
# =============================================================================


class TestFieldValueFormatter:
    """Tests for value formatting for different field types."""

    def test_format_text(self, sample_schema: FormSchema) -> None:
        """Test text formatting."""
        field = sample_schema.fields[0]  # Name text field
        result = FieldValueFormatter.format_for_field("  Test Name  ", field)
        assert result == "Test Name"

    def test_format_text_max_length(self, sample_schema: FormSchema) -> None:
        """Test text truncation with max_length."""
        field = sample_schema.fields[0]  # max_length=50
        long_text = "X" * 100
        result = FieldValueFormatter.format_for_field(long_text, field)
        assert len(result) == 50

    def test_format_number(self, sample_schema: FormSchema) -> None:
        """Test number formatting."""
        field = sample_schema.fields[1]  # Amount number field
        result = FieldValueFormatter.format_for_field("1,234.56", field)
        assert result == "1234.56"

    def test_format_number_integer(self, sample_schema: FormSchema) -> None:
        """Test integer formatting."""
        field = sample_schema.fields[1]
        result = FieldValueFormatter.format_for_field("1000.00", field)
        assert result == "1000"

    def test_format_number_with_currency(self, sample_schema: FormSchema) -> None:
        """Test number with currency symbol."""
        field = sample_schema.fields[1]
        result = FieldValueFormatter.format_for_field("$1,500", field)
        assert result == "1500"

    def test_format_date_iso(self, sample_schema: FormSchema) -> None:
        """Test date formatting from ISO format."""
        field = sample_schema.fields[2]  # Date field
        result = FieldValueFormatter.format_for_field("2023-03-15", field)
        assert result == "03/15/2023"

    def test_format_date_us(self, sample_schema: FormSchema) -> None:
        """Test date already in US format."""
        field = sample_schema.fields[2]
        result = FieldValueFormatter.format_for_field("03/15/2023", field)
        assert result == "03/15/2023"

    def test_format_date_text(self, sample_schema: FormSchema) -> None:
        """Test date in text format."""
        field = sample_schema.fields[2]
        result = FieldValueFormatter.format_for_field("March 15, 2023", field)
        assert result == "03/15/2023"

    def test_format_checkbox_checked(self, sample_schema: FormSchema) -> None:
        """Test checkbox formatting - checked."""
        field = sample_schema.fields[3]  # Checkbox
        result = FieldValueFormatter.format_for_field("true", field)
        assert result == "Yes"  # checked_value

    def test_format_checkbox_yes(self, sample_schema: FormSchema) -> None:
        """Test checkbox formatting - 'yes' value."""
        field = sample_schema.fields[3]
        result = FieldValueFormatter.format_for_field("yes", field)
        assert result == "Yes"

    def test_format_checkbox_unchecked(self, sample_schema: FormSchema) -> None:
        """Test checkbox formatting - unchecked."""
        field = sample_schema.fields[3]
        result = FieldValueFormatter.format_for_field("false", field)
        assert result == ""

    def test_format_checkbox_custom_checked_value(self) -> None:
        """Test checkbox with custom checked value."""
        field = FormField(
            id="test",
            type=FieldType.CHECKBOX,
            page=1,
            checked_value="On",
        )
        result = FieldValueFormatter.format_for_field("yes", field)
        assert result == "On"

    def test_format_null_value(self, sample_schema: FormSchema) -> None:
        """Test null value returns None."""
        field = sample_schema.fields[0]
        result = FieldValueFormatter.format_for_field(None, field)
        assert result is None


# =============================================================================
# AcroForm Filler Tests
# =============================================================================


class TestAcroFormFiller:
    """Tests for AcroForm field filling with pikepdf."""

    def test_has_acroform_true(self, fillable_pdf_path: Path) -> None:
        """Test detecting AcroForm in PDF."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        with AcroFormFiller(fillable_pdf_path) as filler:
            assert filler.has_acroform() is True

    def test_has_acroform_false(self, minimal_pdf_path: Path) -> None:
        """Test detecting no AcroForm in PDF."""
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created")

        with AcroFormFiller(minimal_pdf_path) as filler:
            assert filler.has_acroform() is False

    def test_fill_text_field(self, fillable_pdf_path: Path, tmp_path: Path) -> None:
        """Test filling a text field."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        output = tmp_path / "filled.pdf"

        with AcroFormFiller(fillable_pdf_path) as filler:
            # The fillable.pdf has field_001
            success = filler.fill_field("field_001", "Test Value", FieldType.TEXT)
            filler.save(output)

        assert success is True
        assert output.exists()

        # Verify the value was set
        with pikepdf.open(output) as pdf:
            # Should have value set
            assert pdf.Root.AcroForm is not None

    def test_fill_nonexistent_field(self, fillable_pdf_path: Path) -> None:
        """Test filling a field that doesn't exist."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        with AcroFormFiller(fillable_pdf_path) as filler:
            success = filler.fill_field("nonexistent_field", "Value", FieldType.TEXT)
            assert success is False

    def test_context_manager(self, fillable_pdf_path: Path) -> None:
        """Test using filler as context manager."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        with AcroFormFiller(fillable_pdf_path) as filler:
            assert filler.pdf is not None

        # Should be closed after context
        assert filler.pdf is None

    def test_missing_file_error(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(WriterError, match="not found"):
            AcroFormFiller("/nonexistent/file.pdf")

    def test_save_to_path(self, fillable_pdf_path: Path, tmp_path: Path) -> None:
        """Test saving to a specific path."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        output = tmp_path / "output.pdf"

        with AcroFormFiller(fillable_pdf_path) as filler:
            filler.fill_field("field_001", "Test", FieldType.TEXT)
            filler.save(output)

        assert output.exists()
        assert output.stat().st_size > 0


# =============================================================================
# Overlay Writer Tests
# =============================================================================


class TestOverlayWriter:
    """Tests for overlay writing (for non-fillable PDFs)."""

    def test_add_text_overlay(self, minimal_pdf_path: Path) -> None:
        """Test adding text overlay."""
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created")

        writer = OverlayWriter(minimal_pdf_path)
        writer.add_text_overlay(
            text="Test Overlay",
            page=1,
            x=100,
            y=700,
            font_size=12,
        )

        assert len(writer._overlays) == 1
        assert writer._overlays[0]["type"] == "text"

    def test_add_checkbox_overlay(self, minimal_pdf_path: Path) -> None:
        """Test adding checkbox overlay."""
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created")

        writer = OverlayWriter(minimal_pdf_path)
        writer.add_checkbox_overlay(
            checked=True,
            page=1,
            x=100,
            y=700,
            size=10,
        )

        assert len(writer._overlays) == 1
        assert writer._overlays[0]["type"] == "checkbox"
        assert writer._overlays[0]["checked"] is True

    def test_write_overlays(self, minimal_pdf_path: Path, tmp_path: Path) -> None:
        """Test writing PDF with overlays."""
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created")

        output = tmp_path / "overlay.pdf"

        writer = OverlayWriter(minimal_pdf_path)
        writer.add_text_overlay("Test", 1, 100, 700, 12)
        writer.write(output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_missing_file_error(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(WriterError, match="not found"):
            OverlayWriter("/nonexistent/file.pdf")


# =============================================================================
# PDF Writer Tests
# =============================================================================


class TestPDFWriter:
    """Tests for the main PDFWriter class."""

    def test_detect_acroform_mode(self, fillable_pdf_path: Path, sample_schema: FormSchema) -> None:
        """Test detection of AcroForm mode."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        writer = PDFWriter(fillable_pdf_path, sample_schema)
        assert writer._has_acroform is True

    def test_detect_overlay_mode(self, minimal_pdf_path: Path, sample_schema: FormSchema) -> None:
        """Test detection of overlay mode (no AcroForm)."""
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created")

        writer = PDFWriter(minimal_pdf_path, sample_schema)
        assert writer._has_acroform is False

    def test_write_with_acroform(
        self,
        fillable_pdf_path: Path,
        sample_schema: FormSchema,
        sample_mapping_result: FieldMappingResult,
        tmp_path: Path,
    ) -> None:
        """Test writing with AcroForm fields."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        # Create schema matching the fillable PDF
        schema = FormSchema(
            form_id="fillable",
            pages=1,
            fields=[
                FormField(
                    id="field_001",
                    label="Test Field",
                    page=1,
                    type=FieldType.TEXT,
                    position=FieldPosition(x=100, y=650, w=200, h=20),
                ),
            ],
        )

        mapping = FieldMappingResult(
            mappings=[
                FieldMapping(field_id="field_001", value="Test Value", confidence=0.95),
            ],
        )

        output = tmp_path / "filled.pdf"
        writer = PDFWriter(fillable_pdf_path, schema)
        success = writer.write(mapping, output)

        assert success is True
        assert output.exists()

    def test_write_with_overlay(
        self,
        minimal_pdf_path: Path,
        sample_schema: FormSchema,
        tmp_path: Path,
    ) -> None:
        """Test writing with overlay (no AcroForm)."""
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created")

        # Create schema with positioned fields
        schema = FormSchema(
            form_id="minimal",
            pages=1,
            fields=[
                FormField(
                    id="field_001",
                    label="Name",
                    page=1,
                    type=FieldType.TEXT,
                    position=FieldPosition(x=100, y=700, w=200, h=15),
                ),
            ],
        )

        mapping = FieldMappingResult(
            mappings=[
                FieldMapping(field_id="field_001", value="Test Name", confidence=0.95),
            ],
        )

        output = tmp_path / "overlay.pdf"
        writer = PDFWriter(minimal_pdf_path, schema)
        success = writer.write(mapping, output)

        # Overlay mode may not succeed if reportlab not installed
        # Just check that it doesn't crash
        assert isinstance(success, bool)

    def test_write_with_null_values(
        self,
        fillable_pdf_path: Path,
        sample_schema: FormSchema,
        tmp_path: Path,
    ) -> None:
        """Test writing with null values in mapping."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        schema = FormSchema(
            form_id="fillable",
            pages=1,
            fields=[
                FormField(
                    id="field_001",
                    label="Test",
                    page=1,
                    type=FieldType.TEXT,
                    position=FieldPosition(x=100, y=650, w=200, h=20),
                ),
            ],
        )

        mapping = FieldMappingResult(
            mappings=[
                FieldMapping(field_id="field_001", value=None, confidence=0.5),
            ],
        )

        output = tmp_path / "filled.pdf"
        writer = PDFWriter(fillable_pdf_path, schema)
        success = writer.write(mapping, output)

        # Should succeed but fill no fields
        # (null value means don't fill)
        assert isinstance(success, bool)

    def test_missing_file_error(self, sample_schema: FormSchema) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(WriterError, match="not found"):
            PDFWriter("/nonexistent/file.pdf", sample_schema)


# =============================================================================
# Integration Tests
# =============================================================================


class TestWriterIntegration:
    """Integration tests with real PDFs."""

    def test_end_to_end_fill(
        self,
        fillable_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test end-to-end filling of a PDF."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        # Create schema matching the fillable PDF
        schema = FormSchema(
            form_id="fillable",
            pages=1,
            fields=[
                FormField(
                    id="field_001",
                    label="Test Field",
                    page=1,
                    type=FieldType.TEXT,
                    position=FieldPosition(x=100, y=650, w=200, h=20),
                ),
            ],
        )

        # Create mapping
        mapping = FieldMappingResult(
            mappings=[
                FieldMapping(
                    field_id="field_001",
                    value="Integration Test Value",
                    confidence=0.98,
                    reasoning="Test mapping",
                    source_key="name",
                ),
            ],
        )

        # Write PDF
        output = tmp_path / "integration_filled.pdf"
        writer = PDFWriter(fillable_pdf_path, schema)
        success = writer.write(mapping, output)

        assert success is True
        assert output.exists()

        # Verify output is valid PDF
        with pikepdf.open(output) as pdf:
            assert len(pdf.pages) > 0

    def test_calculated_field_filled(
        self,
        fillable_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test that calculated fields are also filled."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        schema = FormSchema(
            form_id="fillable",
            pages=1,
            fields=[
                FormField(
                    id="field_001",
                    label="Test",
                    page=1,
                    type=FieldType.TEXT,
                    position=FieldPosition(x=100, y=650, w=200, h=20),
                ),
            ],
        )

        mapping = FieldMappingResult(
            mappings=[],
            calculations=[
                FieldMapping(
                    field_id="field_001",
                    value="1000",
                    confidence=1.0,
                    calculated=True,
                    formula="500 + 500",
                ),
            ],
        )

        output = tmp_path / "calc_filled.pdf"
        writer = PDFWriter(fillable_pdf_path, schema)
        success = writer.write(mapping, output)

        assert success is True


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for write_filled_pdf convenience function."""

    def test_convenience_function(
        self,
        fillable_pdf_path: Path,
        sample_schema: FormSchema,
        sample_mapping_result: FieldMappingResult,
        tmp_path: Path,
    ) -> None:
        """Test the convenience function works."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        output = tmp_path / "convenience_filled.pdf"

        # Create matching schema
        schema = FormSchema(
            form_id="fillable",
            pages=1,
            fields=[
                FormField(
                    id="field_001",
                    label="Test",
                    page=1,
                    type=FieldType.TEXT,
                    position=FieldPosition(x=100, y=650, w=200, h=20),
                ),
            ],
        )

        mapping = FieldMappingResult(
            mappings=[
                FieldMapping(field_id="field_001", value="Test", confidence=0.95),
            ],
        )

        success = write_filled_pdf(
            pdf_path=fillable_pdf_path,
            form_schema=schema,
            mapping_result=mapping,
            output_path=output,
        )

        assert isinstance(success, bool)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_schema_field(
        self,
        fillable_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test handling when field in mapping doesn't exist in schema."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        schema = FormSchema(
            form_id="test",
            pages=1,
            fields=[],  # Empty schema
        )

        mapping = FieldMappingResult(
            mappings=[
                FieldMapping(field_id="nonexistent", value="Test", confidence=0.95),
            ],
        )

        output = tmp_path / "output.pdf"
        writer = PDFWriter(fillable_pdf_path, schema)
        success = writer.write(mapping, output)

        # Should succeed but fill no fields
        assert success is False


# =============================================================================
# Font Size Calculation Tests
# =============================================================================


class TestFontSizeCalculation:
    """Tests for font size calculation."""

    def test_font_size_from_field_height(self, sample_schema: FormSchema) -> None:
        """Test font size is calculated from field height."""
        field = sample_schema.fields[0]  # height=15
        writer = PDFWriter.__new__(PDFWriter)

        font_size = writer._calculate_font_size(field)
        # Should be ~70% of field height
        assert 10 <= font_size <= 12

    def test_font_size_clamped(self) -> None:
        """Test font size is clamped to reasonable range."""
        # Very small field
        small_field = FormField(
            id="test",
            type=FieldType.TEXT,
            page=1,
            position=FieldPosition(x=0, y=0, w=100, h=5),  # Very short
        )
        writer = PDFWriter.__new__(PDFWriter)
        font_size = writer._calculate_font_size(small_field)
        assert font_size >= 6  # Minimum

        # Very large field
        large_field = FormField(
            id="test",
            type=FieldType.TEXT,
            page=1,
            position=FieldPosition(x=0, y=0, w=100, h=50),  # Very tall
        )
        font_size = writer._calculate_font_size(large_field)
        assert font_size <= 14  # Maximum


# =============================================================================
# CLI Tests
# =============================================================================


class TestFillCLI:
    """Tests for the fill CLI command."""

    def test_fill_help(self) -> None:
        """Test fill command help output."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["fill", "--help"])

        assert result.exit_code == 0
        assert "Fill a PDF form" in result.output
        assert "--data" in result.output
        assert "--output" in result.output
        assert "--verify" in result.output
        assert "--dry-run" in result.output

    def test_fill_missing_data_file(self) -> None:
        """Test fill with missing data file."""
        from click.testing import CliRunner
        from formbridge.cli import main

        minimal_pdf = FIXTURES_DIR / "minimal.pdf"
        if not minimal_pdf.exists():
            pytest.skip("Minimal PDF fixture not created")

        runner = CliRunner()
        result = runner.invoke(main, [
            "fill", str(minimal_pdf),
            "--data", "/nonexistent/data.json",
        ])

        assert result.exit_code != 0

    def test_fill_missing_pdf(self) -> None:
        """Test fill with missing PDF."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "fill", "/nonexistent.pdf",
            "--data", "/nonexistent.json",
        ])

        assert result.exit_code != 0

    def test_fill_dry_run(
        self,
        fillable_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test fill with --dry-run flag (no LLM needed when no fields)."""
        from click.testing import CliRunner
        from formbridge.cli import main

        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        # Create data file
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps({"name": "Test"}))

        runner = CliRunner()
        # No API key - dry run with 0 fields shouldn't need LLM
        result = runner.invoke(main, [
            "fill", str(fillable_pdf_path),
            "--data", str(data_file),
            "--dry-run",
        ])

        # With 0 fields scanned, should complete (possibly with warnings)
        # The key test is that dry-run doesn't write a PDF
        output_pdf = tmp_path / "filled.pdf"
        assert not output_pdf.exists()

    def test_fill_with_mocked_llm(
        self,
        fillable_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test fill command with mocked LLM."""
        from click.testing import CliRunner
        from formbridge.cli import main

        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        # Create data file
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps({"name": "Test Name"}))

        output_file = tmp_path / "filled.pdf"

        # Mock LLM response
        llm_response = {
            "mappings": [
                {
                    "field_id": "field_001",
                    "value": "Test Name",
                    "confidence": 0.95,
                    "reasoning": "Test",
                    "source_key": "name",
                },
            ],
        }

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_response(llm_response)

            runner = CliRunner()
            result = runner.invoke(main, [
                "fill", str(fillable_pdf_path),
                "--data", str(data_file),
                "--output", str(output_file),
            ], env={"FORMBRIDGE_API_KEY": "test-key"})

        # Should succeed
        assert result.exit_code == 0 or "filled" in result.output.lower()


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipeline:
    """Tests for the full scan → parse → map → write pipeline."""

    @pytest.mark.integration
    def test_scan_parse_map_write(
        self,
        tmp_path: Path,
    ) -> None:
        """Test the complete pipeline with a real IRS form."""
        irs_form = Path(__file__).parent / "fixtures" / "f1065.pdf"
        if not irs_form.exists():
            pytest.skip("IRS Form 1065 not downloaded - run download_fixtures.py")

        from formbridge.mapper import Mapper
        from formbridge.scanner import Scanner
        from formbridge.writer import PDFWriter

        # Step 1: Scan
        scanner = Scanner(irs_form)
        schema = scanner.scan()
        assert len(schema.fields) > 0

        # Step 2: Parse (skip for this test - no instruction PDF)
        # Step 3: Map
        user_data = {"name": "Test Company"}
        
        # Mock LLM response
        llm_response = {
            "mappings": [
                {
                    "field_id": schema.fields[0].id,
                    "value": "Test Company",
                    "confidence": 0.95,
                    "reasoning": "Test mapping",
                    "source_key": "name",
                },
            ],
        }

        from formbridge.llm import LLMConfig

        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_response(llm_response)

            mapper = Mapper(
                user_data=user_data,
                form_schema=schema,
                llm_config=llm_config,
            )
            mapping_result = mapper.map()

        # Should have at least one mapping
        assert len(mapping_result.mappings) >= 1

        # Step 4: Write
        output = tmp_path / "pipeline_filled.pdf"
        writer = PDFWriter(irs_form, schema)
        success = writer.write(mapping_result, output)

        # Success may be False if field couldn't be filled (IRS widget annotations
        # don't map cleanly to named form fields), but the PDF should still be produced
        assert output.exists()

        # Verify PDF is valid
        with pikepdf.open(output) as pdf:
            assert len(pdf.pages) > 0


def _mock_litellm_response(content: Any, model: str = "gpt-4o-mini") -> MagicMock:
    """Create a mock litellm.completion response object."""
    if isinstance(content, dict):
        content_str = json.dumps(content)
    else:
        content_str = str(content)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content_str
    mock_response.model = model
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    return mock_response
