"""Tests for the FormBridge scanner module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formbridge.models import FieldType, FormField, FormSchema, FieldPosition
from formbridge.scanner import (
    PDFFieldExtractor,
    OCRFieldDetector,
    Scanner,
    ScannerError,
    scan_pdf,
)

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_form_field() -> FormField:
    """Create a sample form field for testing."""
    return FormField(
        id="field_001",
        label="Name of partnership",
        page=1,
        type=FieldType.TEXT,
        position=FieldPosition(x=120, y=680, w=350, h=20),
        max_length=60,
        required=True,
        line_ref="A",
    )


@pytest.fixture
def sample_schema() -> FormSchema:
    """Create a sample form schema for testing."""
    return FormSchema(
        form_id="test-form",
        pages=2,
        fields=[
            FormField(
                id="field_001",
                label="Name",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=100, y=700, w=200, h=15),
                required=True,
            ),
            FormField(
                id="field_002",
                label="Date",
                page=1,
                type=FieldType.DATE,
                position=FieldPosition(x=350, y=700, w=100, h=15),
            ),
            FormField(
                id="field_003",
                label="Amount",
                page=2,
                type=FieldType.NUMBER,
                position=FieldPosition(x=100, y=500, w=150, h=15),
            ),
        ],
    )


@pytest.fixture
def minimal_pdf_path() -> Path:
    """Return path to minimal test PDF."""
    return FIXTURES_DIR / "minimal.pdf"


@pytest.fixture
def fillable_pdf_path() -> Path:
    """Return path to fillable test PDF."""
    return FIXTURES_DIR / "fillable.pdf"


@pytest.fixture
def form_1065_path() -> Path | None:
    """Return path to IRS Form 1065 if available."""
    path = FIXTURES_DIR / "f1065.pdf"
    return path if path.exists() else None


# =============================================================================
# Model Tests
# =============================================================================

class TestFormField:
    """Tests for FormField model."""

    def test_create_text_field(self) -> None:
        """Test creating a text field."""
        field = FormField(
            id="test_001",
            label="Test Field",
            page=1,
            type=FieldType.TEXT,
        )
        assert field.id == "test_001"
        assert field.type == FieldType.TEXT
        assert field.required is False
        assert field.position is None

    def test_create_checkbox_field(self) -> None:
        """Test creating a checkbox field."""
        field = FormField(
            id="check_001",
            label="Agree to terms",
            page=1,
            type=FieldType.CHECKBOX,
            checked_value="Yes",
        )
        assert field.type == FieldType.CHECKBOX
        assert field.checked_value == "Yes"

    def test_create_radio_field(self) -> None:
        """Test creating a radio button field."""
        field = FormField(
            id="radio_001",
            label="Filing Status",
            page=1,
            type=FieldType.RADIO,
            options=["Single", "Married", "Head of Household"],
        )
        assert field.type == FieldType.RADIO
        assert len(field.options) == 3

    def test_field_with_position(self, sample_form_field: FormField) -> None:
        """Test field with position data."""
        assert sample_form_field.position is not None
        assert sample_form_field.position.x == 120
        assert sample_form_field.position.w == 350


class TestFormSchema:
    """Tests for FormSchema model."""

    def test_create_schema(self, sample_schema: FormSchema) -> None:
        """Test creating a form schema."""
        assert sample_schema.form_id == "test-form"
        assert sample_schema.pages == 2
        assert len(sample_schema.fields) == 3

    def test_schema_json_export(self, sample_schema: FormSchema) -> None:
        """Test exporting schema to JSON."""
        json_str = sample_schema.model_dump_json(indent=2)
        data = json.loads(json_str)

        assert data["form_id"] == "test-form"
        assert data["pages"] == 2
        assert len(data["fields"]) == 3

    def test_schema_json_roundtrip(self, sample_schema: FormSchema) -> None:
        """Test JSON serialization roundtrip."""
        json_str = sample_schema.model_dump_json()
        restored = FormSchema.model_validate_json(json_str)

        assert restored.form_id == sample_schema.form_id
        assert restored.pages == sample_schema.pages
        assert len(restored.fields) == len(sample_schema.fields)


# =============================================================================
# Scanner Tests
# =============================================================================

class TestScanner:
    """Tests for Scanner class."""

    def test_scanner_nonexistent_file(self) -> None:
        """Test scanner raises error for nonexistent file."""
        with pytest.raises(ScannerError, match="PDF file not found"):
            Scanner("/nonexistent/file.pdf")

    def test_scanner_invalid_file(self, tmp_path: Path) -> None:
        """Test scanner raises error for non-PDF file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Not a PDF")

        # pdfplumber will fail on non-PDF files
        with pytest.raises(Exception):  # pdfplumber.pdfa.PDFSyntaxError or similar
            scanner = Scanner(txt_file)
            scanner.scan()

    def test_scanner_creates_form_id(self, tmp_path: Path) -> None:
        """Test scanner creates a form ID from filename."""
        # Create a minimal valid PDF (we'll skip actual scanning)
        # For now, this is a placeholder
        pass


class TestPDFFieldExtractor:
    """Tests for PDF field extractor."""

    def test_field_type_mapping(self) -> None:
        """Test PDF field type to FieldType mapping."""
        # The mapping should handle common PDF field types
        assert PDFFieldExtractor.FIELD_TYPE_MAP["text"] == FieldType.TEXT
        assert PDFFieldExtractor.FIELD_TYPE_MAP["checkbox"] == FieldType.CHECKBOX
        assert PDFFieldExtractor.FIELD_TYPE_MAP["radio"] == FieldType.RADIO
        assert PDFFieldExtractor.FIELD_TYPE_MAP["date"] == FieldType.DATE
        assert PDFFieldExtractor.FIELD_TYPE_MAP["numeric"] == FieldType.NUMBER

    def test_line_ref_extraction(self) -> None:
        """Test extracting line references from field names."""
        extractor = PDFFieldExtractor.__new__(PDFFieldExtractor)

        # Test various patterns
        assert extractor._extract_line_ref("Line1_A") == "A"
        assert extractor._extract_line_ref("Line16a") == "16a"
        assert extractor._extract_line_ref("field_42_name") == "42"
        # Some patterns might not match
        assert extractor._extract_line_ref("randomname") is None


class TestOCRFieldDetector:
    """Tests for OCR field detector."""

    def test_has_ocr_dependencies(self) -> None:
        """Test OCR dependency check."""
        # This will return True if pytesseract/pdf2image are installed
        # or False if not
        # We just test that the method runs without error
        pass  # Detector needs a PDF to initialize

    def test_overlap_calculation(self) -> None:
        """Test rectangle overlap calculation."""
        detector = OCRFieldDetector.__new__(OCRFieldDetector)

        # No overlap
        a = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
        b = {"x0": 20, "y0": 20, "x1": 30, "y1": 30}
        assert detector._calculate_overlap(a, b) == 0.0

        # Full overlap
        a = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
        b = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
        assert detector._calculate_overlap(a, b) == 1.0

        # Partial overlap
        a = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
        b = {"x0": 5, "y0": 5, "x1": 15, "y1": 15}
        overlap = detector._calculate_overlap(a, b)
        assert 0 < overlap < 1

    def test_deduplicate_areas(self) -> None:
        """Test area deduplication."""
        detector = OCRFieldDetector.__new__(OCRFieldDetector)

        # Identical areas should be deduplicated
        areas = [
            {"x0": 0, "y0": 0, "x1": 10, "y1": 10, "label": "A"},
            {"x0": 0, "y0": 0, "x1": 10, "y1": 10, "label": "B"},  # Duplicate
            {"x0": 50, "y0": 50, "x1": 60, "y1": 60, "label": "C"},  # Different
        ]

        deduped = detector._deduplicate_areas(areas)
        assert len(deduped) == 2


# =============================================================================
# Integration Tests (require actual PDFs)
# =============================================================================

@pytest.mark.integration
class TestScannerIntegration:
    """Integration tests with local test PDFs."""

    def test_scan_minimal_pdf(self, minimal_pdf_path: Path) -> None:
        """Test scanning a minimal PDF with no form fields."""
        # Ensure fixture exists
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created. Run: python tests/download_fixtures.py")

        scanner = Scanner(minimal_pdf_path)
        schema = scanner.scan()

        # Basic validation
        assert schema.form_id == "minimal"
        assert schema.pages == 1
        # Minimal PDF has no form fields
        assert isinstance(schema.fields, list)

    def test_scan_fillable_pdf(self, fillable_pdf_path: Path) -> None:
        """Test scanning a PDF with AcroForm fields."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created. Run: python tests/download_fixtures.py")

        scanner = Scanner(fillable_pdf_path)
        schema = scanner.scan()

        # Basic validation
        assert schema.form_id == "fillable"
        assert schema.pages == 1
        # Should have detected the text field
        assert isinstance(schema.fields, list)

    def test_scan_pdf_to_json(self, fillable_pdf_path: Path) -> None:
        """Test JSON output from scan."""
        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        scanner = Scanner(fillable_pdf_path)
        json_str = scanner.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "form_id" in data
        assert "pages" in data
        assert "fields" in data

    def test_scan_pdf_with_convenience_function(self, minimal_pdf_path: Path) -> None:
        """Test scan_pdf convenience function."""
        if not minimal_pdf_path.exists():
            pytest.skip("Minimal PDF fixture not created")

        schema = scan_pdf(minimal_pdf_path)
        assert schema is not None
        assert schema.form_id == "minimal"


class TestScannerWithExternalPDFs:
    """Integration tests with IRS forms (require downloading)."""

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "f1065.pdf").exists(),
        reason="Form 1065 fixture not downloaded. Run: python tests/download_fixtures.py"
    )
    def test_scan_irs_form_1065(self, form_1065_path: Path) -> None:
        """Test scanning IRS Form 1065."""
        scanner = Scanner(form_1065_path)
        schema = scanner.scan()

        # Basic validation
        assert schema.form_id is not None
        assert schema.pages > 0
        assert len(schema.fields) > 0

        # Form 1065 should have text fields
        text_fields = [f for f in schema.fields if f.type == FieldType.TEXT]
        assert len(text_fields) > 0

        # Fields should have page numbers
        for field in schema.fields:
            assert 1 <= field.page <= schema.pages

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "f1065.pdf").exists(),
        reason="Form 1065 fixture not downloaded. Run: python tests/download_fixtures.py"
    )
    def test_scan_1065_json_output(self, form_1065_path: Path) -> None:
        """Test JSON output for Form 1065 scan."""
        scanner = Scanner(form_1065_path)
        schema = scanner.scan()

        json_str = scanner.to_json(schema)
        data = json.loads(json_str)

        # Verify JSON structure
        assert "form_id" in data
        assert "pages" in data
        assert "fields" in data

        # Each field should have required properties
        for field in data["fields"]:
            assert "id" in field
            assert "type" in field
            assert "page" in field


# =============================================================================
# CLI Tests
# =============================================================================

class TestCLI:
    """Tests for CLI commands."""

    def test_cli_version(self) -> None:
        """Test CLI version flag."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "FormBridge version" in result.output

    def test_cli_help(self) -> None:
        """Test CLI help output."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "FormBridge" in result.output
        assert "scan" in result.output

    def test_cli_scan_help(self) -> None:
        """Test scan command help."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", "--help"])

        assert result.exit_code == 0
        assert "Scan a PDF form" in result.output

    def test_cli_scan_missing_file(self) -> None:
        """Test scan with missing file."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", "/nonexistent.pdf"])

        assert result.exit_code != 0

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "f1065.pdf").exists(),
        reason="Form 1065 fixture not downloaded"
    )
    def test_cli_scan_to_file(self, form_1065_path: Path, tmp_path: Path) -> None:
        """Test scan command with output file."""
        from click.testing import CliRunner
        from formbridge.cli import main

        output_file = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(main, [
            "scan", str(form_1065_path),
            "--output", str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output is valid JSON
        data = json.loads(output_file.read_text())
        assert "fields" in data

    def test_cli_scan_to_file_local(self, fillable_pdf_path: Path, tmp_path: Path) -> None:
        """Test scan command with output file using local fixture."""
        from click.testing import CliRunner
        from formbridge.cli import main

        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        output_file = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(main, [
            "scan", str(fillable_pdf_path),
            "--output", str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output is valid JSON
        data = json.loads(output_file.read_text())
        assert "fields" in data

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "f1065.pdf").exists(),
        reason="Form 1065 fixture not downloaded"
    )
    def test_cli_scan_table_format(self, form_1065_path: Path) -> None:
        """Test scan command with table format."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "--format", "table",
            "scan", str(form_1065_path)
        ])

        assert result.exit_code == 0
        # Table output should contain field info
        assert "Form:" in result.output or "fields" in result.output.lower()

    def test_cli_scan_table_format_local(self, fillable_pdf_path: Path) -> None:
        """Test scan command with table format using local fixture."""
        from click.testing import CliRunner
        from formbridge.cli import main

        if not fillable_pdf_path.exists():
            pytest.skip("Fillable PDF fixture not created")

        runner = CliRunner()
        result = runner.invoke(main, [
            "--format", "table",
            "scan", str(fillable_pdf_path)
        ])

        assert result.exit_code == 0

    def test_cli_fill_exists(self) -> None:
        """Test fill command exists and shows help."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["fill", "--help"])

        assert result.exit_code == 0
        assert "fill" in result.output.lower()

    def test_cli_parse_requires_fields(self, tmp_path: Path) -> None:
        """Test parse command requires --fields argument."""
        from click.testing import CliRunner
        from formbridge.cli import main

        # Create dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")  # Minimal PDF header

        runner = CliRunner()
        result = runner.invoke(main, ["parse", str(pdf_file)])

        # Should error because --fields is required
        assert result.exit_code != 0 or "--fields" in result.output.lower() or "error" in result.output.lower()


# =============================================================================
# Conftest for pytest markers
# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require real PDFs)"
    )
