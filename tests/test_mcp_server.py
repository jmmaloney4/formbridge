"""Tests for FormBridge MCP Server."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formbridge.mcp_server import (
    TOOLS,
    call_tool,
    handle_fill,
    handle_scan,
    handle_template_create,
    handle_templates,
    handle_verify,
    list_tools,
    server,
)
from formbridge.models import (
    FieldType,
    FormField,
    FormSchema,
    FieldMapping,
    FieldMappingResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_form_schema() -> FormSchema:
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
                required=True,
                line_ref="A",
            ),
            FormField(
                id="field_002",
                label="EIN",
                page=1,
                type=FieldType.TEXT,
                required=True,
                line_ref="B",
            ),
        ],
    )


@pytest.fixture
def sample_mapping_result() -> FieldMappingResult:
    """Create a sample mapping result for testing."""
    return FieldMappingResult(
        mappings=[
            FieldMapping(
                field_id="field_001",
                value="Test Corp",
                confidence=0.95,
                reasoning="Direct match with user data",
                source_key="name",
            ),
            FieldMapping(
                field_id="field_002",
                value="12-3456789",
                confidence=0.99,
                reasoning="EIN format match",
                source_key="ein",
            ),
        ],
        unmapped_fields=[],
        unmapped_data=[],
        calculations=[],
        warnings=[],
    )


@pytest.fixture
def temp_pdf(tmp_path: Path) -> Path:
    """Create a minimal test PDF file."""
    pdf_path = tmp_path / "test.pdf"
    # Minimal PDF content (just a valid PDF header and footer)
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Test MCP tool registration."""

    def test_list_tools_returns_all_tools(self):
        """Test that list_tools returns all defined tools."""
        import asyncio
        tools = asyncio.run(list_tools())
        assert len(tools) == 5

    def test_tools_have_required_fields(self):
        """Test that all tools have required fields."""
        for tool in TOOLS:
            assert tool.name, "Tool must have a name"
            assert tool.description, "Tool must have a description"
            assert tool.inputSchema, "Tool must have an inputSchema"
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"

    def test_tool_names_are_correct(self):
        """Test that tool names match expected values."""
        expected_names = {
            "formbridge_scan",
            "formbridge_fill",
            "formbridge_verify",
            "formbridge_templates",
            "formbridge_template_create",
        }
        actual_names = {tool.name for tool in TOOLS}
        assert actual_names == expected_names

    def test_scan_tool_has_pdf_path_param(self):
        """Test scan tool has required pdf_path parameter."""
        scan_tool = next(t for t in TOOLS if t.name == "formbridge_scan")
        assert "pdf_path" in scan_tool.inputSchema["properties"]
        assert "pdf_path" in scan_tool.inputSchema["required"]

    def test_fill_tool_has_required_params(self):
        """Test fill tool has all required parameters."""
        fill_tool = next(t for t in TOOLS if t.name == "formbridge_fill")
        props = fill_tool.inputSchema["properties"]
        required = fill_tool.inputSchema["required"]

        assert "form" in props and "form" in required
        assert "data" in props and "data" in required
        assert "output_path" in props and "output_path" in required
        assert "instructions_path" in props  # optional
        assert "dry_run" in props  # optional

    def test_server_instance_created(self):
        """Test that MCP server instance is created correctly."""
        assert server is not None
        assert server.name == "formbridge"


# =============================================================================
# Scan Handler Tests
# =============================================================================


class TestHandleScan:
    """Test formbridge_scan tool handler."""

    @pytest.mark.asyncio
    async def test_scan_missing_pdf_path(self):
        """Test scan with missing pdf_path parameter."""
        result = await handle_scan({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "pdf_path is required" in result[0].text

    @pytest.mark.asyncio
    async def test_scan_nonexistent_file(self):
        """Test scan with non-existent file."""
        result = await handle_scan({"pdf_path": "/nonexistent/file.pdf"})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_scan_success(self, temp_pdf: Path, sample_form_schema: FormSchema):
        """Test successful scan."""
        with patch("formbridge.mcp_server.Scanner") as mock_scanner_class:
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = sample_form_schema
            mock_scanner_class.return_value = mock_scanner

            result = await handle_scan({"pdf_path": str(temp_pdf)})

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert data["success"] is True
            assert data["form_id"] == "test-form"
            assert data["fields_count"] == 2
            assert len(data["fields"]) == 2

    @pytest.mark.asyncio
    async def test_scan_returns_field_structure(
        self, temp_pdf: Path, sample_form_schema: FormSchema
    ):
        """Test that scan returns proper field structure."""
        with patch("formbridge.mcp_server.Scanner") as mock_scanner_class:
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = sample_form_schema
            mock_scanner_class.return_value = mock_scanner

            result = await handle_scan({"pdf_path": str(temp_pdf)})
            data = json.loads(result[0].text)

            field = data["fields"][0]
            assert "id" in field
            assert "label" in field
            assert "page" in field
            assert "type" in field


# =============================================================================
# Fill Handler Tests
# =============================================================================


class TestHandleFill:
    """Test formbridge_fill tool handler."""

    @pytest.mark.asyncio
    async def test_fill_missing_form(self):
        """Test fill with missing form parameter."""
        result = await handle_fill({"data": {}, "output_path": "out.pdf"})
        assert "Error" in result[0].text
        assert "form is required" in result[0].text

    @pytest.mark.asyncio
    async def test_fill_missing_data(self):
        """Test fill with missing data parameter."""
        result = await handle_fill({"form": "test.pdf", "output_path": "out.pdf"})
        assert "Error" in result[0].text
        assert "data is required" in result[0].text

    @pytest.mark.asyncio
    async def test_fill_missing_output_path(self):
        """Test fill with missing output_path parameter."""
        result = await handle_fill({"form": "test.pdf", "data": {"name": "Test"}})
        assert "Error" in result[0].text
        assert "output_path is required" in result[0].text

    @pytest.mark.asyncio
    async def test_fill_dry_run_no_output_needed(self, temp_pdf: Path):
        """Test that dry_run doesn't require output_path."""
        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.exists.return_value = False
            mock_tm_class.return_value = mock_tm

            with patch("formbridge.mcp_server.Scanner") as mock_scanner_class:
                with patch("formbridge.mcp_server.Path.exists", return_value=True):
                    mock_scanner = MagicMock()
                    mock_scanner.scan.return_value = FormSchema(
                        form_id="test", pages=1, fields=[]
                    )
                    mock_scanner_class.return_value = mock_scanner

                    with patch("formbridge.mcp_server.Mapper") as mock_mapper_class:
                        mock_mapper = MagicMock()
                        mock_mapper.map.return_value = FieldMappingResult()
                        mock_mapper_class.return_value = mock_mapper

                        with patch("formbridge.mcp_server.load_llm_config"):
                            result = await handle_fill({
                                "form": str(temp_pdf),
                                "data": {"name": "Test"},
                                "dry_run": True,
                            })

                            assert len(result) == 1
                            data = json.loads(result[0].text)
                            assert data.get("dry_run") is True

    @pytest.mark.asyncio
    async def test_fill_nonexistent_file(self):
        """Test fill with non-existent file (not a template)."""
        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.exists.return_value = False
            mock_tm_class.return_value = mock_tm

            result = await handle_fill({
                "form": "/nonexistent/file.pdf",
                "data": {},
                "output_path": "out.pdf",
            })

            assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_fill_returns_mapping_structure(
        self, temp_pdf: Path, sample_form_schema: FormSchema, sample_mapping_result: FieldMappingResult
    ):
        """Test that fill returns proper mapping structure."""
        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.exists.return_value = False
            mock_tm_class.return_value = mock_tm

            with patch("formbridge.mcp_server.Scanner") as mock_scanner_class:
                with patch("formbridge.mcp_server.Path.exists", return_value=True):
                    mock_scanner = MagicMock()
                    mock_scanner.scan.return_value = sample_form_schema
                    mock_scanner_class.return_value = mock_scanner

                    with patch("formbridge.mcp_server.Mapper") as mock_mapper_class:
                        mock_mapper = MagicMock()
                        mock_mapper.map.return_value = sample_mapping_result
                        mock_mapper_class.return_value = mock_mapper

                        with patch("formbridge.mcp_server.PDFWriter") as mock_writer_class:
                            mock_writer = MagicMock()
                            mock_writer.write.return_value = True
                            mock_writer_class.return_value = mock_writer

                            with patch("formbridge.mcp_server.load_llm_config"):
                                result = await handle_fill({
                                    "form": str(temp_pdf),
                                    "data": {"name": "Test Corp"},
                                    "output_path": str(temp_pdf.parent / "output.pdf"),
                                    "dry_run": True,
                                })

                                data = json.loads(result[0].text)
                                assert "mapping" in data
                                assert "fields_mapped" in data["mapping"]


# =============================================================================
# Verify Handler Tests
# =============================================================================


class TestHandleVerify:
    """Test formbridge_verify tool handler."""

    @pytest.mark.asyncio
    async def test_verify_missing_pdf_path(self):
        """Test verify with missing pdf_path parameter."""
        result = await handle_verify({})
        assert "Error" in result[0].text
        assert "pdf_path is required" in result[0].text

    @pytest.mark.asyncio
    async def test_verify_nonexistent_file(self):
        """Test verify with non-existent file."""
        result = await handle_verify({"pdf_path": "/nonexistent/file.pdf"})
        assert "Error" in result[0].text
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_verify_success(self, temp_pdf: Path):
        """Test successful verify without template."""
        result = await handle_verify({"pdf_path": str(temp_pdf)})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["success"] is True
        assert "pdf_path" in data

    @pytest.mark.asyncio
    async def test_verify_with_template(self, temp_pdf: Path, sample_form_schema: FormSchema):
        """Test verify with template."""
        from formbridge.templates import Template, TemplateManifest

        mock_manifest = TemplateManifest(
            name="test-template",
            display_name="Test Template",
            fields_count=2,
            pages=1,
            created_at="2026-01-01T00:00:00Z",
        )
        mock_template = MagicMock()
        mock_template.manifest = mock_manifest
        mock_template.schema = sample_form_schema
        mock_template.instruction_map = None

        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.get.return_value = mock_template
            mock_tm_class.return_value = mock_tm

            result = await handle_verify({
                "pdf_path": str(temp_pdf),
                "template": "test-template",
            })

            data = json.loads(result[0].text)
            assert data["success"] is True
            assert "form" in data
            assert data["form"]["form_id"] == "test-form"

    @pytest.mark.asyncio
    async def test_verify_template_not_found(self, temp_pdf: Path):
        """Test verify with non-existent template."""
        from formbridge.templates import TemplateNotFoundError

        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.get.side_effect = TemplateNotFoundError("Not found")
            mock_tm_class.return_value = mock_tm

            result = await handle_verify({
                "pdf_path": str(temp_pdf),
                "template": "nonexistent",
            })

            assert "Error" in result[0].text


# =============================================================================
# Templates Handler Tests
# =============================================================================


class TestHandleTemplates:
    """Test formbridge_templates tool handler."""

    @pytest.mark.asyncio
    async def test_templates_list_empty(self):
        """Test templates list when empty."""
        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.list.return_value = []
            mock_tm_class.return_value = mock_tm

            result = await handle_templates({})

            data = json.loads(result[0].text)
            assert data["success"] is True
            assert data["count"] == 0
            assert data["templates"] == []

    @pytest.mark.asyncio
    async def test_templates_list_success(self):
        """Test templates list with templates."""
        from formbridge.templates import TemplateManifest

        mock_manifest = TemplateManifest(
            name="test-template",
            display_name="Test Template",
            fields_count=10,
            pages=2,
            created_at="2026-01-01T00:00:00Z",
        )

        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.list.return_value = [mock_manifest]
            mock_tm_class.return_value = mock_tm

            result = await handle_templates({})

            data = json.loads(result[0].text)
            assert data["success"] is True
            assert data["count"] == 1
            assert len(data["templates"]) == 1
            assert data["templates"][0]["name"] == "test-template"


# =============================================================================
# Template Create Handler Tests
# =============================================================================


class TestHandleTemplateCreate:
    """Test formbridge_template_create tool handler."""

    @pytest.mark.asyncio
    async def test_create_missing_form_pdf(self):
        """Test template create with missing form_pdf parameter."""
        result = await handle_template_create({"name": "test"})
        assert "Error" in result[0].text
        assert "form_pdf is required" in result[0].text

    @pytest.mark.asyncio
    async def test_create_missing_name(self, temp_pdf: Path):
        """Test template create with missing name parameter."""
        result = await handle_template_create({"form_pdf": str(temp_pdf)})
        assert "Error" in result[0].text
        assert "name is required" in result[0].text

    @pytest.mark.asyncio
    async def test_create_nonexistent_file(self):
        """Test template create with non-existent file."""
        result = await handle_template_create({
            "form_pdf": "/nonexistent/file.pdf",
            "name": "test",
        })
        assert "Error" in result[0].text
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_create_success(self, temp_pdf: Path, sample_form_schema: FormSchema):
        """Test successful template creation."""
        from formbridge.templates import Template, TemplateManifest

        mock_manifest = TemplateManifest(
            name="test-template",
            display_name="Test Template",
            fields_count=2,
            pages=1,
            created_at="2026-01-01T00:00:00Z",
        )
        mock_template = MagicMock()
        mock_template.manifest = mock_manifest

        with patch("formbridge.mcp_server.TemplateManager") as mock_tm_class:
            mock_tm = MagicMock()
            mock_tm.create.return_value = mock_template
            mock_tm_class.return_value = mock_tm

            result = await handle_template_create({
                "form_pdf": str(temp_pdf),
                "name": "test-template",
            })

            data = json.loads(result[0].text)
            assert data["success"] is True
            assert "template" in data
            assert data["template"]["name"] == "test-template"


# =============================================================================
# Call Tool Tests
# =============================================================================


class TestCallTool:
    """Test call_tool routing."""

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        """Test calling unknown tool."""
        result = await call_tool("unknown_tool", {})
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_routes_to_handler(self, temp_pdf: Path):
        """Test that call_tool routes to correct handler."""
        with patch("formbridge.mcp_server.handle_scan") as mock_handle:
            mock_handle.return_value = [MagicMock(text="test")]

            await call_tool("formbridge_scan", {"pdf_path": str(temp_pdf)})

            mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_handles_exception(self):
        """Test that call_tool handles exceptions gracefully."""
        with patch("formbridge.mcp_server.handle_scan") as mock_handle:
            mock_handle.side_effect = Exception("Test error")

            result = await call_tool("formbridge_scan", {"pdf_path": "test.pdf"})

            assert "Error" in result[0].text


# =============================================================================
# Stdio Transport Tests
# =============================================================================


class TestStdioTransport:
    """Test stdio transport initialization."""

    def test_run_stdio_exists(self):
        """Test that run_stdio function exists."""
        from formbridge.mcp_server import run_stdio
        assert callable(run_stdio)

    def test_run_http_exists(self):
        """Test that run_http function exists."""
        from formbridge.mcp_server import run_http
        assert callable(run_http)

    def test_main_exists(self):
        """Test that main function exists."""
        from formbridge.mcp_server import main
        assert callable(main)
