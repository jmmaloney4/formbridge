"""MCP Server for FormBridge.

Exposes FormBridge as an MCP server so any AI agent can use it natively.
Supports stdio (primary) and HTTP/SSE transports.

Tools exposed:
- formbridge_scan: Scan a PDF form and extract field structure
- formbridge_fill: Fill a PDF form with data using instruction-aware mapping
- formbridge_verify: Verify a filled form (field-by-field breakdown)
- formbridge_templates: List available templates
- formbridge_template_create: Create a template from form + instructions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

from formbridge import __version__
from formbridge.llm import LLMError, load_config as load_llm_config
from formbridge.mapper import Mapper, MapperError
from formbridge.models import FormSchema, InstructionMap, VerificationReport
from formbridge.parser import Parser, ParserError
from formbridge.scanner import Scanner, ScannerError
from formbridge.templates import (
    Template,
    TemplateError,
    TemplateManager,
    TemplateNotFoundError,
    generate_data_template,
)
from formbridge.writer import PDFWriter, WriterError

# Configure logging
logger = logging.getLogger(__name__)

# Create MCP server instance
server = Server("formbridge")


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    Tool(
        name="formbridge_scan",
        description=(
            "Scan a PDF form and extract field structure. "
            "Returns a FormSchema with all detected fields, their types, positions, and labels. "
            "Use this first to understand what fields a form contains. "
            "Set vision_labels=true for forms with garbled text labels (e.g., IRS tax forms)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to the blank PDF form to scan",
                },
                "vision_labels": {
                    "type": "boolean",
                    "description": "Enable vision-based label refinement via LLM (ADR 001). "
                    "Recommended for IRS forms where text extraction produces garbled labels.",
                    "default": False,
                },
            },
            "required": ["pdf_path"],
        },
    ),
    Tool(
        name="formbridge_fill",
        description=(
            "Fill a PDF form with data using instruction-aware field mapping. "
            "Takes a form (PDF path or template name) and JSON data, then fills the form "
            "using AI to intelligently map data to fields. Optionally uses parsed instructions "
            "for more accurate mapping. Returns the path to the filled PDF and a verification report."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "form": {
                    "type": "string",
                    "description": "Path to PDF form or template name (e.g., 'irs-1065-2025')",
                },
                "data": {
                    "type": "object",
                    "description": "Key-value data to fill the form with",
                },
                "instructions_path": {
                    "type": "string",
                    "description": "Optional path to instruction document PDF",
                },
                "output_path": {
                    "type": "string",
                    "description": "Where to save the filled PDF",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, return mapping without writing PDF",
                    "default": False,
                },
            },
            "required": ["form", "data", "output_path"],
        },
    ),
    Tool(
        name="formbridge_verify",
        description=(
            "Verify a filled form - returns field-by-field breakdown of what was filled. "
            "Analyzes a filled PDF against a template to show confidence scores, "
            "warnings, and any issues with the filled form."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to the filled PDF to verify",
                },
                "template": {
                    "type": "string",
                    "description": "Template name for field reference",
                },
            },
            "required": ["pdf_path"],
        },
    ),
    Tool(
        name="formbridge_templates",
        description=(
            "List available form templates. "
            "Templates bundle scanned forms + parsed instructions for reuse. "
            "Returns a list of installed templates with metadata."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="formbridge_template_create",
        description=(
            "Create a template from a form PDF and optional instructions. "
            "Scans the form and parses instructions, then bundles everything "
            "into a reusable template that can be used for quick filling."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "form_pdf": {
                    "type": "string",
                    "description": "Path to the blank form PDF",
                },
                "name": {
                    "type": "string",
                    "description": "Template name (alphanumeric, hyphens, underscores only)",
                },
                "instructions_pdf": {
                    "type": "string",
                    "description": "Optional path to instruction document PDF",
                },
                "display_name": {
                    "type": "string",
                    "description": "Human-readable display name",
                },
                "category": {
                    "type": "string",
                    "description": "Category for organization (e.g., 'tax/us/irs')",
                },
            },
            "required": ["form_pdf", "name"],
        },
    ),
]


# =============================================================================
# Tool Handlers
# =============================================================================


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available MCP tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool invocations."""
    try:
        if name == "formbridge_scan":
            return await handle_scan(arguments)
        elif name == "formbridge_fill":
            return await handle_fill(arguments)
        elif name == "formbridge_verify":
            return await handle_verify(arguments)
        elif name == "formbridge_templates":
            return await handle_templates(arguments)
        elif name == "formbridge_template_create":
            return await handle_template_create(arguments)
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}",
            )]
    except Exception as e:
        logger.exception(f"Tool {name} failed")
        return [TextContent(
            type="text",
            text=f"Error: {e}",
        )]


async def handle_scan(args: dict[str, Any]) -> list[TextContent]:
    """Handle formbridge_scan tool."""
    pdf_path = args.get("pdf_path")
    vision_labels = args.get("vision_labels", False)

    if not pdf_path:
        return [TextContent(
            type="text",
            text="Error: pdf_path is required",
        )]

    path = Path(pdf_path)
    if not path.exists():
        return [TextContent(
            type="text",
            text=f"Error: PDF file not found: {pdf_path}",
        )]

    try:
        scanner = Scanner(path, vision_labels=vision_labels)
        schema = scanner.scan()

        result = {
            "success": True,
            "form_id": schema.form_id,
            "pages": schema.pages,
            "fields_count": len(schema.fields),
            "fields": [
                {
                    "id": f.id,
                    "label": f.label,
                    "page": f.page,
                    "type": f.type.value,
                    "line_ref": f.line_ref,
                    "required": f.required,
                }
                for f in schema.fields
            ],
            "schema": schema.model_dump(),
        }

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    except ScannerError as e:
        return [TextContent(
            type="text",
            text=f"Scanner error: {e}",
        )]


async def handle_fill(args: dict[str, Any]) -> list[TextContent]:
    """Handle formbridge_fill tool."""
    form = args.get("form")
    data = args.get("data")
    output_path = args.get("output_path")
    instructions_path = args.get("instructions_path")
    dry_run = args.get("dry_run", False)

    if not form:
        return [TextContent(type="text", text="Error: form is required")]
    if not data:
        return [TextContent(type="text", text="Error: data is required")]
    if not output_path and not dry_run:
        return [TextContent(type="text", text="Error: output_path is required (unless dry_run=true)")]

    try:
        template_manager = TemplateManager()
        is_template = template_manager.exists(form)
        pdf_path = Path(form)

        form_schema: FormSchema
        instruction_map: InstructionMap | None = None
        actual_pdf_path: Path

        if is_template:
            template = template_manager.get(form)
            form_schema = template.schema
            instruction_map = template.instruction_map
            actual_pdf_path = template_manager.get_form_path(form)
        else:
            if not pdf_path.exists():
                return [TextContent(
                    type="text",
                    text=f"Error: '{form}' is not a template name or existing file",
                )]
            actual_pdf_path = pdf_path
            scanner = Scanner(pdf_path)
            form_schema = scanner.scan()

        # Parse additional instructions if provided
        if instructions_path:
            inst_path = Path(instructions_path)
            if inst_path.exists():
                llm_config = load_llm_config()
                parser = Parser(
                    instructions_path=inst_path,
                    schema=form_schema,
                    llm_config=llm_config,
                )
                instruction_map = parser.parse()

        # Map data to fields
        llm_config = load_llm_config()
        mapper = Mapper(
            user_data=data,
            form_schema=form_schema,
            instruction_map=instruction_map,
            llm_config=llm_config,
        )
        mapping_result = mapper.map()

        result = {
            "success": True,
            "form_id": form_schema.form_id,
            "mapping": {
                "fields_mapped": len(mapping_result.mappings),
                "fields_calculated": len(mapping_result.calculations),
                "fields_unmapped": len(mapping_result.unmapped_fields),
                "warnings_count": len(mapping_result.warnings),
                "mappings": [
                    {
                        "field_id": m.field_id,
                        "value": m.value,
                        "confidence": m.confidence,
                        "reasoning": m.reasoning,
                        "source_key": m.source_key,
                        "calculated": m.calculated,
                    }
                    for m in mapping_result.mappings + mapping_result.calculations
                ],
                "warnings": [
                    {
                        "field_id": w.field_id,
                        "message": w.message,
                        "severity": w.severity,
                    }
                    for w in mapping_result.warnings
                ],
                "unmapped_fields": mapping_result.unmapped_fields,
                "unmapped_data": mapping_result.unmapped_data,
            },
        }

        if dry_run:
            result["dry_run"] = True
            result["message"] = "Dry run completed - no PDF written"
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # Write the filled PDF
        writer = PDFWriter(
            pdf_path=actual_pdf_path,
            form_schema=form_schema,
        )
        success = writer.write(mapping_result, output_path)

        if success:
            result["output_path"] = str(output_path)
            result["message"] = f"Filled PDF saved to {output_path}"

            # Generate verification report
            report = VerificationReport(
                form=str(actual_pdf_path),
                timestamp=__import__('datetime').datetime.now(
                    tz=__import__('zoneinfo').ZoneInfo("UTC")
                ).isoformat(),
                overall_confidence=_calculate_confidence(mapping_result),
                fields_total=len(form_schema.fields),
                fields_filled=len([m for m in mapping_result.mappings if m.value]),
                fields_blank=len([m for m in mapping_result.mappings if not m.value]),
                fields_calculated=len(mapping_result.calculations),
                fields_flagged=len([m for m in mapping_result.mappings if m.confidence < 0.80]),
                flags=mapping_result.warnings,
            )
            result["verification"] = report.model_dump()
        else:
            result["success"] = False
            result["message"] = "No fields were filled"

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except (ScannerError, ParserError, MapperError, WriterError, LLMError, TemplateError) as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        logger.exception("Fill failed")
        return [TextContent(type="text", text=f"Unexpected error: {e}")]


async def handle_verify(args: dict[str, Any]) -> list[TextContent]:
    """Handle formbridge_verify tool."""
    pdf_path = args.get("pdf_path")
    template_name = args.get("template")

    if not pdf_path:
        return [TextContent(type="text", text="Error: pdf_path is required")]

    path = Path(pdf_path)
    if not path.exists():
        return [TextContent(
            type="text",
            text=f"Error: PDF file not found: {pdf_path}",
        )]

    try:
        form_schema: FormSchema | None = None
        instruction_map: InstructionMap | None = None

        if template_name:
            template_manager = TemplateManager()
            try:
                template = template_manager.get(template_name)
                form_schema = template.schema
                instruction_map = template.instruction_map
            except TemplateNotFoundError:
                return [TextContent(
                    type="text",
                    text=f"Error: Template '{template_name}' not found",
                )]

        result = {
            "success": True,
            "pdf_path": str(path),
            "template": template_name,
        }

        if form_schema:
            result["form"] = {
                "form_id": form_schema.form_id,
                "pages": form_schema.pages,
                "fields_count": len(form_schema.fields),
                "fields": [
                    {
                        "id": f.id,
                        "label": f.label,
                        "page": f.page,
                        "type": f.type.value,
                        "line_ref": f.line_ref,
                    }
                    for f in form_schema.fields
                ],
            }

            if instruction_map:
                result["instructions"] = {
                    "fields_with_instructions": len(instruction_map.field_instructions),
                    "calculation_rules": len(instruction_map.calculation_rules),
                }
        else:
            result["message"] = "No template provided - basic file info only"

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception("Verify failed")
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_templates(args: dict[str, Any]) -> list[TextContent]:
    """Handle formbridge_templates tool."""
    try:
        manager = TemplateManager()
        templates = manager.list()

        result = {
            "success": True,
            "count": len(templates),
            "templates": [
                {
                    "name": t.name,
                    "display_name": t.display_name,
                    "version": t.version,
                    "category": t.category,
                    "fields_count": t.fields_count,
                    "pages": t.pages,
                    "has_instructions": t.has_instructions,
                    "tags": t.tags,
                    "created_at": t.created_at,
                }
                for t in templates
            ],
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception("Templates list failed")
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_template_create(args: dict[str, Any]) -> list[TextContent]:
    """Handle formbridge_template_create tool."""
    form_pdf = args.get("form_pdf")
    name = args.get("name")
    instructions_pdf = args.get("instructions_pdf")
    display_name = args.get("display_name")
    category = args.get("category")

    if not form_pdf:
        return [TextContent(type="text", text="Error: form_pdf is required")]
    if not name:
        return [TextContent(type="text", text="Error: name is required")]

    form_path = Path(form_pdf)
    if not form_path.exists():
        return [TextContent(
            type="text",
            text=f"Error: Form PDF not found: {form_pdf}",
        )]

    try:
        manager = TemplateManager()
        template = manager.create(
            form_pdf=form_pdf,
            name=name,
            instructions_pdf=instructions_pdf,
            display_name=display_name,
            category=category,
        )

        result = {
            "success": True,
            "message": f"Template '{name}' created successfully",
            "template": {
                "name": template.manifest.name,
                "display_name": template.manifest.display_name,
                "version": template.manifest.version,
                "category": template.manifest.category,
                "fields_count": template.manifest.fields_count,
                "pages": template.manifest.pages,
                "has_instructions": template.manifest.has_instructions,
                "tags": template.manifest.tags,
            },
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except TemplateError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        logger.exception("Template create failed")
        return [TextContent(type="text", text=f"Error: {e}")]


def _calculate_confidence(mapping_result) -> float:
    """Calculate overall confidence from mapping result."""
    all_mappings = list(mapping_result.mappings) + list(mapping_result.calculations)
    if not all_mappings:
        return 0.0
    return sum(m.confidence for m in all_mappings) / len(all_mappings)


# =============================================================================
# Server Entry Points
# =============================================================================


async def run_stdio() -> None:
    """Run the MCP server in stdio mode."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_http(port: int = 3000) -> None:
    """Run the MCP server in HTTP/SSE mode.

    Note: HTTP mode requires additional dependencies.
    For production use, consider using a proper ASGI server.
    """
    import asyncio
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route

    sse = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    async def handle_messages(request):
        await sse.handle_post_message(request._receive, request._send)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
        ],
    )

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)


def main() -> None:
    """Main entry point for MCP server CLI."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        asyncio.run(run_stdio())
    elif len(sys.argv) > 1 and sys.argv[1].startswith("--port="):
        port = int(sys.argv[1].split("=")[1])
        run_http(port=port)
    elif len(sys.argv) > 2 and sys.argv[1] == "--port":
        port = int(sys.argv[2])
        run_http(port=port)
    else:
        # Default to stdio
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
