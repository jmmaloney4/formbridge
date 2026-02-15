"""Tests for the FormBridge template system (Phase 4).

Covers:
- Template creation from form + instructions
- Template listing
- Template loading
- Template deletion
- Template installation from registry (mocked)
- Data template generation
- Template-based fill workflow
- Verification report generation
- CLI commands for templates
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from formbridge.models import (
    CalculationRule,
    FieldInstruction,
    FieldMapping,
    FieldMappingResult,
    FieldPosition,
    FieldType,
    FormField,
    FormSchema,
    InstructionMap,
    VerificationReport,
)
from formbridge.templates import (
    DEFAULT_TEMPLATES_DIR,
    Template,
    TemplateAlreadyExistsError,
    TemplateError,
    TemplateManifest,
    TemplateManager,
    TemplateNotFoundError,
    generate_data_template,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_templates_dir(tmp_path: Path) -> Path:
    """Create a temporary templates directory."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


@pytest.fixture
def template_manager(temp_templates_dir: Path) -> TemplateManager:
    """Create a TemplateManager with temporary directory."""
    return TemplateManager(templates_dir=temp_templates_dir)


@pytest.fixture
def sample_schema() -> FormSchema:
    """Create a sample form schema."""
    return FormSchema(
        form_id="test-form-1065",
        pages=2,
        fields=[
            FormField(
                id="field_001",
                label="Name of partnership",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=120, y=680, w=350, h=20),
                line_ref="A",
                required=True,
            ),
            FormField(
                id="field_002",
                label="EIN",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=480, y=680, w=120, h=20),
                line_ref="B",
            ),
            FormField(
                id="field_003",
                label="Gross receipts",
                page=1,
                type=FieldType.NUMBER,
                line_ref="1",
            ),
            FormField(
                id="field_004",
                label="Deductions",
                page=1,
                type=FieldType.NUMBER,
                line_ref="2",
            ),
            FormField(
                id="field_005",
                label="Net income",
                page=1,
                type=FieldType.NUMBER,
                line_ref="3",
            ),
        ],
    )


@pytest.fixture
def sample_instruction_map() -> InstructionMap:
    """Create a sample instruction map."""
    return InstructionMap(
        form_id="test-form-1065",
        field_instructions={
            "field_001": FieldInstruction(
                line_ref="A",
                label="Name of partnership",
                instruction="Enter the legal name of the partnership.",
                examples=["Smith & Jones LLP"],
            ),
            "field_002": FieldInstruction(
                line_ref="B",
                label="EIN",
                instruction="Enter the nine-digit EIN.",
                format="XX-XXXXXXX",
            ),
        },
        calculation_rules=[
            CalculationRule(
                target="field_005",
                line_ref="3",
                formula="field_003 - field_004",
                description="Gross receipts minus deductions",
            ),
        ],
    )


@pytest.fixture
def sample_manifest() -> TemplateManifest:
    """Create a sample template manifest."""
    return TemplateManifest(
        name="test-form",
        display_name="Test Form 2025",
        version="1.0.0",
        category="test/category",
        tags=["test", "example"],
        fields_count=5,
        pages=2,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        has_instructions=True,
    )


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal valid PDF file for testing."""
    pdf_path = tmp_path / "test_form.pdf"
    _create_minimal_pdf(pdf_path)
    return pdf_path


@pytest.fixture
def sample_instructions_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal instructions PDF for testing."""
    pdf_path = tmp_path / "test_instructions.pdf"
    _create_minimal_pdf(pdf_path)
    return pdf_path


# =============================================================================
# Helper Functions
# =============================================================================


def _create_minimal_pdf(path: Path) -> None:
    """Create a minimal valid PDF file using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as rl_canvas

        c = rl_canvas.Canvas(str(path), pagesize=letter)
        c.drawString(100, 700, "Test Form")
        c.drawString(100, 680, "Name: _______________")
        c.drawString(100, 660, "EIN: _______________")
        c.save()
        return
    except ImportError:
        pass

    # Fallback: copy from test fixtures if available
    fixtures_dir = Path(__file__).parent / "fixtures"
    minimal = fixtures_dir / "minimal.pdf"
    if minimal.exists():
        import shutil
        shutil.copy(minimal, path)
        return

    # Last resort: handcrafted minimal PDF
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer<</Root 1 0 R/Size 5>>
startxref
307
%%EOF"""
    path.write_bytes(content)


def _mock_httpx_response(status_code: int, json_body: Any = None, text: str = "") -> httpx.Response:
    """Create a mock httpx.Response."""
    request = httpx.Request("GET", "https://mock.api.com/endpoint")
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=request)
    return httpx.Response(status_code, text=text, request=request)


# =============================================================================
# TemplateManifest Tests
# =============================================================================


class TestTemplateManifest:
    """Tests for TemplateManifest model."""

    def test_create_manifest(self) -> None:
        """Test creating a manifest."""
        manifest = TemplateManifest(
            name="test-form",
            display_name="Test Form",
            fields_count=10,
            pages=2,
            created_at="2026-01-01T00:00:00Z",
        )

        assert manifest.name == "test-form"
        assert manifest.display_name == "Test Form"
        assert manifest.version == "1.0.0"
        assert manifest.fields_count == 10
        assert manifest.has_instructions is False

    def test_manifest_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        manifest = TemplateManifest(
            name="test",
            display_name="Test",
            fields_count=5,
            pages=1,
            created_at="2026-01-01T00:00:00Z",
            tags=["a", "b"],
            category="test/cat",
        )

        json_str = manifest.model_dump_json()
        restored = TemplateManifest.model_validate_json(json_str)

        assert restored.name == manifest.name
        assert restored.tags == manifest.tags

    def test_manifest_defaults(self) -> None:
        """Test manifest default values."""
        manifest = TemplateManifest(
            name="test",
            display_name="Test",
            fields_count=1,
            pages=1,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        assert manifest.version == "1.0.0"
        assert manifest.tags == []
        assert manifest.has_instructions is False
        assert manifest.source_url is None


# =============================================================================
# TemplateManager Tests
# =============================================================================


class TestTemplateManager:
    """Tests for TemplateManager class."""

    def test_manager_initialization(self, temp_templates_dir: Path) -> None:
        """Test manager initializes correctly."""
        manager = TemplateManager(templates_dir=temp_templates_dir)

        assert manager.templates_dir == temp_templates_dir
        assert temp_templates_dir.exists()

    def test_manager_creates_directory(self, tmp_path: Path) -> None:
        """Test manager creates templates directory if it doesn't exist."""
        templates_dir = tmp_path / "new_templates"
        assert not templates_dir.exists()

        manager = TemplateManager(templates_dir=templates_dir)

        assert templates_dir.exists()

    def test_is_valid_name(self, template_manager: TemplateManager) -> None:
        """Test template name validation."""
        assert template_manager._is_valid_name("test-form")
        assert template_manager._is_valid_name("irs_1065_2025")
        assert template_manager._is_valid_name("Form123")
        assert not template_manager._is_valid_name("test form")  # Space
        assert not template_manager._is_valid_name("test.form")  # Dot
        assert not template_manager._is_valid_name("")  # Empty

    def test_list_empty(self, template_manager: TemplateManager) -> None:
        """Test listing when no templates exist."""
        templates = template_manager.list()
        assert templates == []

    def test_exists_false(self, template_manager: TemplateManager) -> None:
        """Test exists returns False for non-existent template."""
        assert template_manager.exists("nonexistent") is False


class TestTemplateCreate:
    """Tests for template creation."""

    def test_create_template_without_instructions(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test creating a template without instructions."""
        template = template_manager.create(
            form_pdf=sample_pdf_path,
            name="test-template",
            display_name="Test Template",
        )

        assert template is not None
        assert template.name == "test-template"
        assert template.manifest.display_name == "Test Template"
        assert template.schema.fields is not None  # Schema should exist (may have 0 fields for simple PDFs)
        assert template.has_instructions is False

        # Verify files were created
        template_dir = template_manager.templates_dir / "test-template"
        assert template_dir.exists()
        assert (template_dir / "manifest.json").exists()
        assert (template_dir / "schema.json").exists()
        assert (template_dir / "form.pdf").exists()

    def test_create_template_already_exists(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test error when template already exists."""
        # Create first template
        template_manager.create(form_pdf=sample_pdf_path, name="existing")

        # Try to create again
        with pytest.raises(TemplateAlreadyExistsError, match="already exists"):
            template_manager.create(form_pdf=sample_pdf_path, name="existing")

    def test_create_template_invalid_name(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test error with invalid template name."""
        with pytest.raises(TemplateError, match="Invalid template name"):
            template_manager.create(form_pdf=sample_pdf_path, name="invalid name!")

    def test_create_template_missing_pdf(
        self,
        template_manager: TemplateManager,
    ) -> None:
        """Test error when PDF doesn't exist."""
        with pytest.raises(TemplateError, match="not found"):
            template_manager.create(form_pdf="/nonexistent.pdf", name="test")

    def test_create_template_with_category_and_tags(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test creating template with category and tags."""
        template = template_manager.create(
            form_pdf=sample_pdf_path,
            name="tagged-template",
            category="tax/us/irs",
            tags=["1065", "partnership", "2025"],
        )

        assert template.manifest.category == "tax/us/irs"
        assert template.manifest.tags == ["1065", "partnership", "2025"]


class TestTemplateList:
    """Tests for template listing."""

    def test_list_templates(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test listing templates."""
        # Create some templates
        template_manager.create(form_pdf=sample_pdf_path, name="template-a")
        template_manager.create(form_pdf=sample_pdf_path, name="template-b")

        templates = template_manager.list()

        assert len(templates) == 2
        names = [t.name for t in templates]
        assert "template-a" in names
        assert "template-b" in names

    def test_list_sorted_by_name(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test templates are sorted by name."""
        template_manager.create(form_pdf=sample_pdf_path, name="zebra")
        template_manager.create(form_pdf=sample_pdf_path, name="alpha")
        template_manager.create(form_pdf=sample_pdf_path, name="middle")

        templates = template_manager.list()

        names = [t.name for t in templates]
        assert names == ["alpha", "middle", "zebra"]


class TestTemplateGet:
    """Tests for loading templates."""

    def test_get_template(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test getting a template."""
        # Create template first
        template_manager.create(
            form_pdf=sample_pdf_path,
            name="test-get",
            display_name="Test Get Template",
        )

        # Get it back
        template = template_manager.get("test-get")

        assert template is not None
        assert template.name == "test-get"
        assert template.manifest.display_name == "Test Get Template"
        assert template.schema is not None

    def test_get_nonexistent_template(
        self,
        template_manager: TemplateManager,
    ) -> None:
        """Test error when getting non-existent template."""
        with pytest.raises(TemplateNotFoundError, match="not found"):
            template_manager.get("nonexistent")

    def test_get_template_with_corrupted_manifest(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test error when manifest is corrupted."""
        # Create template
        template_manager.create(form_pdf=sample_pdf_path, name="corrupted")

        # Corrupt the manifest
        manifest_path = template_manager.templates_dir / "corrupted" / "manifest.json"
        manifest_path.write_text("not valid json {")

        with pytest.raises(TemplateError, match="Failed to load manifest"):
            template_manager.get("corrupted")


class TestTemplateDelete:
    """Tests for template deletion."""

    def test_delete_template(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test deleting a template."""
        # Create template
        template_manager.create(form_pdf=sample_pdf_path, name="to-delete")

        # Verify it exists
        assert template_manager.exists("to-delete")

        # Delete it
        result = template_manager.delete("to-delete")

        assert result is True
        assert not template_manager.exists("to-delete")

        # Verify directory is gone
        template_dir = template_manager.templates_dir / "to-delete"
        assert not template_dir.exists()

    def test_delete_nonexistent_template(
        self,
        template_manager: TemplateManager,
    ) -> None:
        """Test error when deleting non-existent template."""
        with pytest.raises(TemplateNotFoundError, match="not found"):
            template_manager.delete("nonexistent")


# =============================================================================
# Template Install Tests (Mocked)
# =============================================================================


class TestTemplateInstall:
    """Tests for template installation from registry."""

    def test_install_template_mocked(
        self,
        template_manager: TemplateManager,
        sample_schema: FormSchema,
        sample_instruction_map: InstructionMap,
    ) -> None:
        """Test installing a template from registry (mocked HTTP)."""
        # Mock responses
        manifest_data = {
            "name": "irs-1065-2025",
            "display_name": "IRS Form 1065 (2025)",
            "version": "1.0.0",
            "category": "tax/us/irs",
            "tags": ["irs", "1065"],
            "fields_count": 5,
            "pages": 2,
            "has_instructions": True,
            "created_at": "2026-01-01T00:00:00Z",
        }

        schema_data = sample_schema.model_dump()
        instructions_data = sample_instruction_map.model_dump()

        def mock_get(url: str, **kwargs):
            if "manifest.json" in url:
                return _mock_httpx_response(200, json_body=manifest_data)
            elif "schema.json" in url:
                return _mock_httpx_response(200, json_body=schema_data)
            elif "instructions.json" in url:
                return _mock_httpx_response(200, json_body=instructions_data)
            elif "form.pdf" in url:
                # Return minimal PDF bytes
                return httpx.Response(
                    200,
                    content=b"%PDF-1.4\ntest",
                    request=httpx.Request("GET", url),
                )
            return _mock_httpx_response(404)

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            template = template_manager.install("irs-1065-2025")

        assert template is not None
        assert template.name == "irs-1065-2025"
        assert template.manifest.display_name == "IRS Form 1065 (2025)"
        assert len(template.schema.fields) == 5

    def test_install_already_exists(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test error when template already installed."""
        # Create local template
        template_manager.create(form_pdf=sample_pdf_path, name="existing")

        with pytest.raises(TemplateAlreadyExistsError, match="already installed"):
            template_manager.install("existing")

    def test_install_not_found_in_registry(
        self,
        template_manager: TemplateManager,
    ) -> None:
        """Test error when template not in registry."""
        def mock_get(url: str, **kwargs):
            return _mock_httpx_response(404)

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with pytest.raises((TemplateNotFoundError, TemplateError)):
                template_manager.install("nonexistent")


# =============================================================================
# Data Template Generator Tests
# =============================================================================


class TestDataTemplateGenerator:
    """Tests for data template generation."""

    def test_generate_data_template(
        self,
        sample_schema: FormSchema,
    ) -> None:
        """Test generating a data template."""
        template = Template(
            path=Path("/tmp/test"),
            manifest=TemplateManifest(
                name="test",
                display_name="Test",
                fields_count=len(sample_schema.fields),
                pages=sample_schema.pages,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
            ),
            schema=sample_schema,
        )

        data_template = generate_data_template(template)

        assert isinstance(data_template, dict)
        # Should have keys based on field labels
        assert "name_of_partnership" in data_template or "line_a" in data_template
        # Should have field descriptions
        assert "_field_descriptions" in data_template

    def test_generate_data_template_with_instructions(
        self,
        sample_schema: FormSchema,
        sample_instruction_map: InstructionMap,
    ) -> None:
        """Test generating data template includes instruction hints."""
        template = Template(
            path=Path("/tmp/test"),
            manifest=TemplateManifest(
                name="test",
                display_name="Test",
                fields_count=len(sample_schema.fields),
                pages=sample_schema.pages,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                has_instructions=True,
            ),
            schema=sample_schema,
            instruction_map=sample_instruction_map,
        )

        data_template = generate_data_template(template)

        # Should include descriptions from instructions
        assert "_field_descriptions" in data_template

    def test_field_to_key_conversion(self) -> None:
        """Test field to key conversion logic."""
        from formbridge.templates import _field_to_key

        field = FormField(
            id="field_001",
            label="Name of Partnership",
            page=1,
            type=FieldType.TEXT,
            line_ref="A",
        )

        key = _field_to_key(field)
        assert key == "name_of_partnership"

        # Field with only line ref
        field2 = FormField(
            id="field_002",
            label=None,
            page=1,
            type=FieldType.NUMBER,
            line_ref="1",
        )

        key2 = _field_to_key(field2)
        assert key2 == "line_1"

    def test_placeholder_generation(self) -> None:
        """Test placeholder value generation."""
        from formbridge.templates import _generate_placeholder

        template = Template(
            path=Path("/tmp/test"),
            manifest=TemplateManifest(
                name="test",
                display_name="Test",
                fields_count=1,
                pages=1,
                created_at="2026-01-01T00:00:00Z",
            ),
            schema=FormSchema(form_id="test", pages=1, fields=[]),
        )

        # Text field
        text_field = FormField(id="f1", type=FieldType.TEXT, page=1)
        assert _generate_placeholder(text_field, template) == ""

        # Number field
        num_field = FormField(id="f2", type=FieldType.NUMBER, page=1)
        assert _generate_placeholder(num_field, template) == 0

        # Date field
        date_field = FormField(id="f3", type=FieldType.DATE, page=1)
        assert _generate_placeholder(date_field, template) == "YYYY-MM-DD"

        # Checkbox field
        check_field = FormField(id="f4", type=FieldType.CHECKBOX, page=1)
        assert _generate_placeholder(check_field, template) is False

        # Radio field with options
        radio_field = FormField(
            id="f5",
            type=FieldType.RADIO,
            page=1,
            options=["A", "B", "C"],
        )
        assert _generate_placeholder(radio_field, template) == "A"


# =============================================================================
# Template Class Tests
# =============================================================================


class TestTemplate:
    """Tests for Template class."""

    def test_template_properties(
        self,
        sample_schema: FormSchema,
        sample_instruction_map: InstructionMap,
        sample_manifest: TemplateManifest,
    ) -> None:
        """Test Template properties."""
        template = Template(
            path=Path("/tmp/test"),
            manifest=sample_manifest,
            schema=sample_schema,
            instruction_map=sample_instruction_map,
        )

        assert template.name == sample_manifest.name
        assert template.has_instructions is True

    def test_template_without_instructions(
        self,
        sample_schema: FormSchema,
        sample_manifest: TemplateManifest,
    ) -> None:
        """Test Template without instructions."""
        template = Template(
            path=Path("/tmp/test"),
            manifest=sample_manifest,
            schema=sample_schema,
            instruction_map=None,
        )

        assert template.has_instructions is False


# =============================================================================
# Verification Report Tests
# =============================================================================


class TestVerificationReport:
    """Tests for verification report generation."""

    def test_verification_report_creation(self) -> None:
        """Test creating a verification report."""
        report = VerificationReport(
            form="test.pdf",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            overall_confidence=0.95,
            fields_total=10,
            fields_filled=8,
            fields_blank=2,
            fields_calculated=2,
            fields_flagged=1,
            flags=[],
        )

        assert report.form == "test.pdf"
        assert report.overall_confidence == 0.95
        assert report.fields_total == 10

    def test_verification_report_json(self) -> None:
        """Test verification report JSON serialization."""
        report = VerificationReport(
            form="test.pdf",
            timestamp="2026-01-01T00:00:00Z",
            overall_confidence=0.95,
            fields_total=10,
            fields_filled=8,
            fields_blank=2,
            fields_calculated=2,
            fields_flagged=1,
            flags=[],
        )

        json_str = report.model_dump_json()
        data = json.loads(json_str)

        assert data["form"] == "test.pdf"
        assert data["overall_confidence"] == 0.95


# =============================================================================
# CLI Tests
# =============================================================================


class TestTemplateCLI:
    """Tests for template CLI commands."""

    def test_template_list_empty(self) -> None:
        """Test template list command when empty."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["template", "list"])

        assert result.exit_code == 0
        assert "No templates installed" in result.output

    def test_template_get_not_found(self) -> None:
        """Test template get command when not found."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["template", "get", "nonexistent"])

        assert result.exit_code != 0

    def test_template_delete_not_found(self) -> None:
        """Test template delete command when not found."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["template", "delete", "nonexistent"])

        assert result.exit_code != 0

    def test_data_template_not_found(self) -> None:
        """Test data-template command when template not found."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["data-template", "nonexistent"])

        assert result.exit_code != 0


class TestVerifyCLI:
    """Tests for verify CLI command."""

    def test_verify_help(self) -> None:
        """Test verify command help."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["verify", "--help"])

        assert result.exit_code == 0
        assert "Verify a filled form" in result.output

    def test_verify_pdf_not_found(self) -> None:
        """Test verify with non-existent PDF."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["verify", "/nonexistent.pdf"])

        assert result.exit_code != 0


class TestFillCLI:
    """Tests for fill CLI command with templates."""

    def test_fill_template_not_found(self) -> None:
        """Test fill with non-existent template."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "fill", "nonexistent-template",
            "--data", "/nonexistent.json",
        ])

        # Should error because template doesn't exist
        assert result.exit_code != 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestTemplateIntegration:
    """Integration tests for the template system."""

    def test_create_list_get_delete_workflow(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test the full create -> list -> get -> delete workflow."""
        # Create
        template = template_manager.create(
            form_pdf=sample_pdf_path,
            name="workflow-test",
            display_name="Workflow Test",
        )
        assert template.name == "workflow-test"

        # List
        templates = template_manager.list()
        assert len(templates) == 1
        assert templates[0].name == "workflow-test"

        # Get
        loaded = template_manager.get("workflow-test")
        assert loaded.name == "workflow-test"

        # Delete
        template_manager.delete("workflow-test")
        assert not template_manager.exists("workflow-test")

        # List empty
        templates = template_manager.list()
        assert len(templates) == 0

    def test_data_template_generation_workflow(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test generating a data template from a form template."""
        # Create template
        template_manager.create(
            form_pdf=sample_pdf_path,
            name="data-test",
        )

        # Generate data template
        template = template_manager.get("data-test")
        data = generate_data_template(template)

        # Should have data keys
        assert isinstance(data, dict)

        # Save to file
        output_path = tmp_path / "data.json"
        output_path.write_text(json.dumps(data, indent=2, default=str))

        assert output_path.exists()

    def test_template_based_fill_workflow(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test filling a form using a template."""
        # Create template
        template = template_manager.create(
            form_pdf=sample_pdf_path,
            name="fill-test",
        )

        # Get the form path
        form_path = template_manager.get_form_path("fill-test")
        assert form_path.exists()

        # Template should be loadable
        loaded = template_manager.get("fill-test")
        assert loaded.schema is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_template_with_no_fields(
        self,
        template_manager: TemplateManager,
        tmp_path: Path,
    ) -> None:
        """Test creating a template from a PDF with no fields."""
        # Create a valid PDF with no form fields
        pdf_path = tmp_path / "empty.pdf"
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as rl_canvas
        c = rl_canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 700, "Empty form")
        c.save()

        # This should work - scanner handles empty PDFs
        template = template_manager.create(
            form_pdf=pdf_path,
            name="empty-test",
        )

        # Should have a schema, possibly with no fields
        assert template.schema is not None

    def test_list_ignores_invalid_templates(
        self,
        template_manager: TemplateManager,
        tmp_path: Path,
    ) -> None:
        """Test that list() skips directories without valid manifests."""
        # Create a directory that looks like a template but isn't
        invalid_dir = template_manager.templates_dir / "invalid-template"
        invalid_dir.mkdir()

        # Should not crash, should return empty list
        templates = template_manager.list()
        assert templates == []

    def test_get_template_missing_schema(
        self,
        template_manager: TemplateManager,
        sample_pdf_path: Path,
    ) -> None:
        """Test error when template is missing schema file."""
        # Create template
        template_manager.create(form_pdf=sample_pdf_path, name="missing-schema")

        # Delete schema file
        schema_path = template_manager.templates_dir / "missing-schema" / "schema.json"
        schema_path.unlink()

        with pytest.raises(TemplateError, match="missing schema"):
            template_manager.get("missing-schema")
