"""FormBridge CLI - Command-line interface for PDF form filling."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from formbridge import __version__
from formbridge.llm import LLMError, load_config as load_llm_config
from formbridge.mapper import Mapper, MapperError
from formbridge.models import (
    FieldMapping,
    FieldMappingResult,
    FieldType,
    FormField,
    FormSchema,
    InstructionMap,
    MappingWarning,
    VerificationReport,
)
from formbridge.parser import Parser, ParserError
from formbridge.scanner import Scanner, ScannerError
from formbridge.templates import (
    Template,
    TemplateAlreadyExistsError,
    TemplateError,
    TemplateManager,
    TemplateNotFoundError,
    generate_data_template,
)
from formbridge.writer import PDFWriter, WriterError

console = Console()


def print_version(ctx: click.Context, param: Any, value: bool) -> None:
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    console.print(f"FormBridge version {__version__}")
    ctx.exit()


def format_output(data: Any, format_type: str) -> str:
    """Format output data based on requested format.

    Args:
        data: Data to format (usually a Pydantic model)
        format_type: Output format (json, csv, table)

    Returns:
        Formatted string
    """
    if format_type == "json":
        if hasattr(data, "model_dump_json"):
            return data.model_dump_json(indent=2)
        return json.dumps(data, indent=2, default=str)

    if format_type == "csv":
        # CSV output for field lists
        if isinstance(data, FormSchema):
            lines = ["id,label,page,type,line_ref,required"]
            for field in data.fields:
                lines.append(
                    f"{field.id},{field.label or ''},{field.page},"
                    f"{field.type.value},{field.line_ref or ''},{field.required}"
                )
            return "\n".join(lines)
        return str(data)

    # Default to table for display
    if isinstance(data, FormSchema):
        table = Table(title=f"Form: {data.form_id}")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Label", style="green")
        table.add_column("Page", justify="right")
        table.add_column("Type", style="yellow")
        table.add_column("Line Ref", style="magenta")

        for field in data.fields:
            table.add_row(
                field.id,
                field.label or "(no label)",
                str(field.page),
                field.type.value,
                field.line_ref or "",
            )

        console.print(table)
        console.print(f"\n[bold]{len(data.fields)}[/] fields across {data.pages} pages")
        return ""

    return str(data)


@click.group()
@click.option("--version", is_flag=True, callback=print_version, expose_value=False, is_eager=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "csv", "table"]), default="json", help="Output format")
@click.option("--provider", envvar="FORMBRIDGE_PROVIDER", help="LLM provider (openai, anthropic, local)")
@click.option("--model", envvar="FORMBRIDGE_MODEL", help="Model name for LLM operations")
@click.pass_context
def main(ctx: click.Context, verbose: bool, output_format: str, provider: str | None, model: str | None) -> None:
    """FormBridge - Instruction-aware PDF form filling.

    Scan, parse, fill, and verify PDF forms with AI assistance.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["format"] = output_format
    ctx.obj["provider"] = provider
    ctx.obj["model"] = model


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for field schema (JSON)")
@click.pass_context
def scan(ctx: click.Context, pdf_path: str, output: str | None) -> None:
    """Scan a PDF form and extract field structure.

    Extracts all fillable fields from a PDF, including field types,
    positions, and labels. Outputs a FormSchema JSON.

    Example:
        formbridge scan form.pdf --output fields.json
    """
    verbose = ctx.obj.get("verbose", False)
    output_format = ctx.obj.get("format", "json")

    try:
        scanner = Scanner(pdf_path, verbose=verbose)
        schema = scanner.scan()

        if output_format == "table":
            format_output(schema, "table")
        elif output:
            output_path = Path(output)
            output_path.write_text(schema.model_dump_json(indent=2))
            console.print(f"[green]✓[/] Schema written to {output}")
        else:
            # Output to stdout
            console.print_json(schema.model_dump_json(indent=2))

    except ScannerError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.argument("instructions_path", type=click.Path(exists=True))
@click.option("--fields", "-f", type=click.Path(exists=True), help="Field schema JSON from scan")
@click.option("--output", "-o", type=click.Path(), help="Output file for instruction map (JSON)")
@click.option("--no-cache", is_flag=True, help="Skip cache and re-parse instructions")
@click.pass_context
def parse(
    ctx: click.Context,
    instructions_path: str,
    fields: str | None,
    output: str | None,
    no_cache: bool,
) -> None:
    """Parse instruction document and map to fields.

    Extracts per-field guidance from instruction documents and maps
    them to field identifiers from a FormSchema.

    Requires a scanned form schema (--fields). Use 'formbridge scan' first.

    Example:
        formbridge parse instructions.pdf --fields fields.json --output instructions.json
    """
    verbose = ctx.obj.get("verbose", False)
    output_format = ctx.obj.get("format", "json")
    provider_name = ctx.obj.get("provider")
    model_name = ctx.obj.get("model")

    try:
        # Load form schema
        if fields:
            schema_data = json.loads(Path(fields).read_text())
            form_schema = FormSchema.model_validate(schema_data)
            if verbose:
                console.print(f"[dim]Loaded form schema with {len(form_schema.fields)} fields[/]")
        else:
            console.print(
                "[red]Error:[/] --fields is required. "
                "Run 'formbridge scan <form.pdf>' first to generate a schema.",
                err=True,
            )
            sys.exit(1)

        # Load LLM config
        llm_config = load_llm_config(provider=provider_name, model=model_name)
        if verbose:
            console.print(
                f"[dim]LLM: {llm_config.provider}/{llm_config.effective_model}[/]"
            )

        # Parse instructions
        parser = Parser(
            instructions_path=instructions_path,
            schema=form_schema,
            llm_config=llm_config,
            use_cache=not no_cache,
            verbose=verbose,
        )

        with console.status("[bold green]Parsing instructions..."):
            instruction_map = parser.parse()

        # Report summary
        n_instructions = len(instruction_map.field_instructions)
        n_calc_rules = len(instruction_map.calculation_rules)
        console.print(
            f"[green]✓[/] Parsed {n_instructions} field instructions, "
            f"{n_calc_rules} calculation rules"
        )

        # Output results
        if output_format == "table":
            _display_instruction_table(instruction_map)
        elif output:
            output_path = Path(output)
            output_path.write_text(instruction_map.model_dump_json(indent=2))
            console.print(f"[green]✓[/] Instruction map written to {output}")
        else:
            console.print_json(instruction_map.model_dump_json(indent=2))

    except (ParserError, LLMError) as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _display_instruction_table(instruction_map) -> None:
    """Display instruction map in table format."""
    from rich.table import Table

    # Field instructions table
    if instruction_map.field_instructions:
        table = Table(title=f"Field Instructions: {instruction_map.form_id}")
        table.add_column("Field ID", style="cyan", no_wrap=True)
        table.add_column("Line", style="magenta", no_wrap=True)
        table.add_column("Label", style="green")
        table.add_column("Instruction Preview", style="yellow")

        for field_id, instruction in instruction_map.field_instructions.items():
            inst_text = instruction.instruction or ""
            preview = inst_text[:60] + "..." if len(inst_text) > 60 else inst_text
            table.add_row(
                field_id,
                instruction.line_ref or "",
                instruction.label or "",
                preview,
            )

        console.print(table)
        console.print(f"\n[bold]{len(instruction_map.field_instructions)}[/] fields with instructions")

    # Calculation rules table
    if instruction_map.calculation_rules:
        calc_table = Table(title="Calculation Rules")
        calc_table.add_column("Target", style="cyan")
        calc_table.add_column("Formula", style="green")
        calc_table.add_column("Description", style="yellow")

        for rule in instruction_map.calculation_rules:
            calc_table.add_row(
                rule.target,
                rule.formula,
                rule.description or "",
            )

        console.print()
        console.print(calc_table)
        console.print(f"\n[bold]{len(instruction_map.calculation_rules)}[/] calculation rules")


@main.command("fill")
@click.argument("form", type=str)
@click.option("--data", "-d", type=click.Path(exists=True), required=True, help="JSON data file to fill form with")
@click.option("--output", "-o", type=click.Path(), required=False, help="Output path for filled PDF")
@click.option("--verify", is_flag=True, help="Interactive verification before writing")
@click.option("--dry-run", is_flag=True, help="Show mapping without writing PDF")
@click.pass_context
def fill_command(
    ctx: click.Context,
    form: str,
    data: str,
    output: str | None,
    verify: bool,
    dry_run: bool,
) -> None:
    """Fill a PDF form with data using AI-powered field mapping.

    Takes a PDF form (or template name) and JSON data, then fills the form using
    instruction-aware field mapping.

    FORM can be either:
    - A path to a PDF file
    - A template name (e.g., 'irs-1065-2025')

    Examples:
        # Fill with a PDF file
        formbridge fill form.pdf --data mydata.json --output filled.pdf

        # Fill using a template (reuses scanned form + parsed instructions)
        formbridge fill irs-1065-2025 --data mydata.json --output filled.pdf

        # With interactive verification
        formbridge fill form.pdf --data mydata.json --verify --output filled.pdf
    """
    verbose = ctx.obj.get("verbose", False)
    output_format = ctx.obj.get("format", "json")
    provider_name = ctx.obj.get("provider")
    model_name = ctx.obj.get("model")

    # Check if form is a template name or a file path
    template_manager = TemplateManager()
    is_template = template_manager.exists(form)
    pdf_path = Path(form)

    if not is_template and not pdf_path.exists():
        console.print(f"[red]Error:[/] '{form}' is not a template name or existing file")
        sys.exit(1)

    try:
        form_schema: FormSchema
        instruction_map: InstructionMap | None = None
        actual_pdf_path: Path

        if is_template:
            # Load from template
            with console.status(f"[bold green]Loading template '{form}'..."):
                template = template_manager.get(form)
                form_schema = template.schema
                instruction_map = template.instruction_map
                actual_pdf_path = template_manager.get_form_path(form)

            console.print(f"[green]✓[/] Loaded template '{form}'")
            console.print(f"  Fields: {len(form_schema.fields)} | Pages: {form_schema.pages}")
            if instruction_map:
                n_inst = len(instruction_map.field_instructions)
                n_calc = len(instruction_map.calculation_rules)
                console.print(f"  Instructions: {n_inst} fields, {n_calc} calculation rules")
        else:
            # Scan the PDF
            actual_pdf_path = pdf_path
            with console.status("[bold green]Scanning PDF..."):
                scanner = Scanner(pdf_path, verbose=verbose)
                form_schema = scanner.scan()
            console.print(f"[green]✓[/] Scanned {len(form_schema.fields)} fields from {form_schema.pages} pages")

        # Load user data
        user_data = json.loads(Path(data).read_text())
        if verbose:
            console.print(f"[dim]Loaded {len(user_data)} data keys from {data}[/]")

        # Map data to fields
        llm_config = load_llm_config(provider=provider_name, model=model_name)
        if verbose:
            console.print(f"[dim]LLM: {llm_config.provider}/{llm_config.effective_model}[/]")

        with console.status("[bold green]Mapping data to fields..."):
            mapper = Mapper(
                user_data=user_data,
                form_schema=form_schema,
                instruction_map=instruction_map,
                llm_config=llm_config,
                verbose=verbose,
            )
            mapping_result = mapper.map()

        # Report mapping summary
        n_mapped = len(mapping_result.mappings)
        n_calc = len(mapping_result.calculations)
        n_warnings = len(mapping_result.warnings)
        console.print(
            f"[green]✓[/] Mapped {n_mapped} fields, {n_calc} calculated, "
            f"{len(mapping_result.unmapped_fields)} unmapped"
        )

        if n_warnings > 0:
            console.print(f"[yellow]⚠[/] {n_warnings} warnings")

        # Handle dry-run mode
        if dry_run:
            console.print("\n[bold]Dry run - mapping result:[/]")
            console.print_json(mapping_result.model_dump_json(indent=2))
            return

        # Interactive verification if requested
        if verify:
            _display_rich_verification(mapping_result, form_schema, instruction_map)

            # Ask user to proceed
            console.print()
            choice = click.prompt(
                "[a]ccept and write PDF / [e]dit mapping / [q]uit",
                type=click.Choice(["a", "e", "q"]),
                default="a",
            )

            if choice == "q":
                console.print("[yellow]Cancelled[/]")
                return
            elif choice == "e":
                console.print("[yellow]Interactive editing not yet implemented[/]")
                console.print("Proceeding with current mapping...")

        # Determine output path
        if output is None:
            output = str(actual_pdf_path.with_stem(f"{actual_pdf_path.stem}_filled"))

        # Write the filled PDF
        with console.status("[bold green]Writing filled PDF..."):
            writer = PDFWriter(
                pdf_path=actual_pdf_path,
                form_schema=form_schema,
                verbose=verbose,
            )
            success = writer.write(mapping_result, output)

        if success:
            console.print(f"[green]✓[/] Filled PDF saved to {output}")

            # Generate and save verification report
            report = _generate_verification_report(
                form_schema, mapping_result, str(actual_pdf_path)
            )
            report_path = Path(output).with_suffix(".verification.json")
            report_path.write_text(report.model_dump_json(indent=2))
            console.print(f"[dim]Verification report saved to {report_path}[/]")
        else:
            console.print("[yellow]⚠[/] No fields were filled")
            console.print("Check that your data keys match the form field names")

    except (MapperError, WriterError, LLMError, TemplateError) as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/] Invalid JSON in data file: {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _display_rich_verification(
    mapping_result: FieldMappingResult,
    form_schema: FormSchema,
    instruction_map: InstructionMap | None = None,
) -> None:
    """Display rich-formatted verification of mappings with color-coded confidence."""
    console.print()

    # Header panel
    header = Panel(
        Text.from_markup(
            f"[bold]FormBridge Fill Verification[/]\n"
            f"Form: [cyan]{form_schema.form_id}[/]\n"
            f"Fields: {len(form_schema.fields)} | Pages: {form_schema.pages}"
        ),
        border_style="blue",
    )
    console.print(header)

    # Build field lookup
    field_lookup = {f.id: f for f in form_schema.fields}

    # Combine mappings and calculations
    all_mappings = list(mapping_result.mappings) + list(mapping_result.calculations)

    # Group by page
    by_page: dict[int, list[FieldMapping]] = {}
    for mapping in all_mappings:
        field = field_lookup.get(mapping.field_id)
        if field:
            if field.page not in by_page:
                by_page[field.page] = []
            by_page[field.page].append(mapping)

    # Display mappings by page
    for page in sorted(by_page.keys()):
        console.print(f"\n[bold cyan]━━━ Page {page} ━━━[/]")

        for mapping in sorted(by_page[page], key=lambda m: field_lookup.get(m.field_id, FormField(id="", page=1, type=FieldType.TEXT)).label or ""):
            field = field_lookup.get(mapping.field_id)
            if not field:
                continue

            # Color-coded confidence indicator
            if mapping.confidence >= 0.95:
                conf_color = "green"
                conf_icon = "✅"
            elif mapping.confidence >= 0.80:
                conf_color = "yellow"
                conf_icon = "⚠️"
            else:
                conf_color = "red"
                conf_icon = "❌"

            # Format value display
            value_display = mapping.value or "[dim](blank)[/]"
            if mapping.value and len(mapping.value) > 35:
                value_display = mapping.value[:32] + "..."

            # Build label/line reference
            if field.line_ref:
                label_text = f"Line {field.line_ref}"
            else:
                label_text = field.label or field.id

            # Main line
            line = Text()
            line.append("  ")
            line.append(conf_icon)
            line.append(" ")
            line.append(f"{label_text}", style="bold")
            line.append(" ... ")
            line.append(str(value_display))
            line.append(" ")
            line.append(f"(conf: {mapping.confidence:.2f})", style=f"{conf_color} dim")

            if mapping.calculated:
                line.append(" ")
                line.append("[calc]", style="cyan")

            console.print(line)

            # Show reasoning for flagged fields
            if mapping.confidence < 0.95 and mapping.reasoning:
                console.print(f"      [dim]↳ {mapping.reasoning}[/]")

    # Show unmapped fields section
    if mapping_result.unmapped_fields:
        console.print(f"\n[bold red]━━━ Unmapped Fields ({len(mapping_result.unmapped_fields)}) ━━━[/]")
        for field_id in mapping_result.unmapped_fields[:10]:
            field = field_lookup.get(field_id)
            if field:
                label = field.label or field_id
                line_ref = f" (Line {field.line_ref})" if field.line_ref else ""
                console.print(f"  • {label}{line_ref}")
        if len(mapping_result.unmapped_fields) > 10:
            console.print(f"  [dim]... and {len(mapping_result.unmapped_fields) - 10} more[/]")

    # Show unused data keys
    if mapping_result.unmapped_data:
        console.print(f"\n[bold yellow]━━━ Unused Data Keys ({len(mapping_result.unmapped_data)}) ━━━[/]")
        for key in mapping_result.unmapped_data[:10]:
            console.print(f"  • {key}")
        if len(mapping_result.unmapped_data) > 10:
            console.print(f"  [dim]... and {len(mapping_result.unmapped_data) - 10} more[/]")

    # Show warnings
    if mapping_result.warnings:
        console.print(f"\n[bold yellow]━━━ Warnings ({len(mapping_result.warnings)}) ━━━[/]")
        for warning in mapping_result.warnings[:10]:
            severity_color = "red" if warning.severity == "error" else "yellow" if warning.severity == "warning" else "blue"
            field_ref = f"[{warning.field_id}] " if warning.field_id else ""
            console.print(f"  [{severity_color}]•[/{severity_color}] {field_ref}{warning.message}")
        if len(mapping_result.warnings) > 10:
            console.print(f"  [dim]... and {len(mapping_result.warnings) - 10} more[/]")

    # Summary panel
    total_fields = len(form_schema.fields)
    filled_fields = len(all_mappings)
    high_conf = len([m for m in all_mappings if m.confidence >= 0.95])
    med_conf = len([m for m in all_mappings if 0.80 <= m.confidence < 0.95])
    low_conf = len([m for m in all_mappings if m.confidence < 0.80])
    overall_conf = _calculate_overall_confidence(all_mappings)

    summary_text = (
        f"Fields mapped: [bold]{filled_fields}/{total_fields}[/]\n"
        f"Overall confidence: [bold]{overall_conf:.2f}[/]\n\n"
        f"[green]✅ High (≥0.95):[/] {high_conf}\n"
        f"[yellow]⚠️ Medium (0.80-0.94):[/] {med_conf}\n"
        f"[red]❌ Low (<0.80):[/] {low_conf}"
    )

    summary_panel = Panel(
        Text.from_markup(summary_text),
        title="[bold]Summary[/]",
        border_style="blue",
    )
    console.print()
    console.print(summary_panel)


def _calculate_overall_confidence(mappings: list[FieldMapping]) -> float:
    """Calculate overall confidence score."""
    if not mappings:
        return 0.0
    return sum(m.confidence for m in mappings) / len(mappings)


def _generate_verification_report(
    form_schema: FormSchema,
    mapping_result: FieldMappingResult,
    form_path: str,
) -> VerificationReport:
    """Generate a verification report."""
    all_mappings = list(mapping_result.mappings) + list(mapping_result.calculations)

    return VerificationReport(
        form=form_path,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        overall_confidence=_calculate_overall_confidence(all_mappings),
        fields_total=len(form_schema.fields),
        fields_filled=len([m for m in all_mappings if m.value is not None]),
        fields_blank=len([m for m in all_mappings if m.value is None]),
        fields_calculated=len(mapping_result.calculations),
        fields_flagged=len([m for m in all_mappings if m.confidence < 0.80]),
        flags=mapping_result.warnings,
    )


# =============================================================================
# Verify Command
# =============================================================================


@main.command("verify")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--template", "-t", "template_name", type=str, help="Template name for field reference")
@click.option("--report", "-r", type=click.Path(), help="Output path for verification report JSON")
@click.pass_context
def verify_command(
    ctx: click.Context,
    pdf_path: str,
    template_name: str | None,
    report: str | None,
) -> None:
    """Verify a filled form - show field-by-field breakdown.

    Analyzes a filled PDF and displays what values are in each field.
    Requires a template to understand the form structure.

    Example:
        formbridge verify filled.pdf --template irs-1065-2025
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load template if provided
        form_schema: FormSchema | None = None
        instruction_map: InstructionMap | None = None

        if template_name:
            template_manager = TemplateManager()
            try:
                template = template_manager.get(template_name)
                form_schema = template.schema
                instruction_map = template.instruction_map
                console.print(f"[green]✓[/] Loaded template '{template_name}'")
            except TemplateNotFoundError:
                console.print(f"[red]Error:[/] Template '{template_name}' not found")
                sys.exit(1)

        # Read the filled PDF
        pdf_file = Path(pdf_path)
        console.print(f"\n[bold]Verifying:[/] {pdf_file.name}")

        if form_schema:
            # Display form structure from template
            console.print(f"[bold]Form:[/] {form_schema.form_id}")
            console.print(f"[bold]Fields:[/] {len(form_schema.fields)}")
            console.print(f"[bold]Pages:[/] {form_schema.pages}")

            if instruction_map:
                n_inst = len(instruction_map.field_instructions)
                n_calc = len(instruction_map.calculation_rules)
                console.print(f"[bold]Instructions:[/] {n_inst} fields, {n_calc} calculation rules")

            # Display field summary
            console.print()
            field_table = Table(title="Form Fields")
            field_table.add_column("ID", style="cyan", no_wrap=True)
            field_table.add_column("Label", style="green")
            field_table.add_column("Page", justify="right")
            field_table.add_column("Type", style="yellow")
            field_table.add_column("Line", style="magenta")

            for field in form_schema.fields[:30]:  # Limit to 30 for display
                field_table.add_row(
                    field.id,
                    (field.label or "")[:30],
                    str(field.page),
                    field.type.value,
                    field.line_ref or "",
                )

            console.print(field_table)

            if len(form_schema.fields) > 30:
                console.print(f"  [dim]... and {len(form_schema.fields) - 30} more fields[/]")
        else:
            console.print("[yellow]No template provided - showing basic PDF info only[/]")
            console.print("Use --template to see field structure")

        # Generate basic report
        basic_report = VerificationReport(
            form=str(pdf_file),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            overall_confidence=0.0,
            fields_total=len(form_schema.fields) if form_schema else 0,
            fields_filled=0,
            fields_blank=0,
            fields_calculated=0,
            fields_flagged=0,
            flags=[],
        )

        # Save report if requested
        if report:
            report_path = Path(report)
            report_path.write_text(basic_report.model_dump_json(indent=2))
            console.print(f"\n[green]✓[/] Report saved to {report}")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


# =============================================================================
# Template Commands
# =============================================================================


@main.group("template")
def template_group() -> None:
    """Manage form templates.

    Templates bundle scanned forms and parsed instructions for reuse.
    They enable scan-once, fill-many workflows.

    Templates are stored in ~/.formbridge/templates/
    """
    pass


@template_group.command("create")
@click.argument("form_pdf", type=click.Path(exists=True))
@click.argument("instructions_pdf", type=click.Path(exists=True), required=False)
@click.option("--name", "-n", required=True, help="Template name (alphanumeric, hyphens, underscores)")
@click.option("--display-name", "-d", help="Human-readable display name")
@click.option("--category", "-c", help="Category (e.g., 'tax/us/irs')")
@click.option("--tag", "-t", multiple=True, help="Tags (can specify multiple)")
@click.pass_context
def template_create(
    ctx: click.Context,
    form_pdf: str,
    instructions_pdf: str | None,
    name: str,
    display_name: str | None,
    category: str | None,
    tag: tuple[str, ...],
) -> None:
    """Create a new template from a form PDF.

    Scans the form and optionally parses instructions, then packages
    everything into a reusable template.

    Example:
        formbridge template create f1065.pdf i1065.pdf --name irs-1065-2025 --category tax/us/irs
    """
    verbose = ctx.obj.get("verbose", False)
    output_format = ctx.obj.get("format", "json")
    provider_name = ctx.obj.get("provider")
    model_name = ctx.obj.get("model")

    try:
        manager = TemplateManager()

        with console.status(f"[bold green]Creating template '{name}'..."):
            template = manager.create(
                form_pdf=form_pdf,
                name=name,
                instructions_pdf=instructions_pdf,
                display_name=display_name,
                category=category,
                tags=list(tag) if tag else None,
                verbose=verbose,
            )

        console.print(f"[green]✓[/] Template '{name}' created successfully")
        console.print(f"  Fields: {template.manifest.fields_count}")
        console.print(f"  Pages: {template.manifest.pages}")
        console.print(f"  Has instructions: {template.manifest.has_instructions}")

        if output_format == "json":
            console.print_json(template.manifest.model_dump_json(indent=2))

    except TemplateAlreadyExistsError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    except TemplateError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@template_group.command("list")
@click.pass_context
def template_list(ctx: click.Context) -> None:
    """List installed templates.

    Example:
        formbridge template list
    """
    verbose = ctx.obj.get("verbose", False)
    output_format = ctx.obj.get("format", "json")

    try:
        manager = TemplateManager()
        templates = manager.list()

        if not templates:
            console.print("[yellow]No templates installed[/]")
            console.print("\nTo create a template:")
            console.print("  formbridge template create form.pdf instructions.pdf --name my-form")
            console.print("\nTo install from registry:")
            console.print("  formbridge template install irs-1065-2025")
            return

        if output_format == "json":
            data = [t.model_dump() for t in templates]
            console.print_json(json.dumps(data, indent=2, default=str))
        else:
            table = Table(title="Installed Templates")
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Display Name", style="green")
            table.add_column("Fields", justify="right")
            table.add_column("Pages", justify="right")
            table.add_column("Instructions", style="yellow")
            table.add_column("Category", style="magenta")

            for t in templates:
                table.add_row(
                    t.name,
                    t.display_name,
                    str(t.fields_count),
                    str(t.pages),
                    "✓" if t.has_instructions else "",
                    t.category or "",
                )

            console.print(table)
            console.print(f"\n[bold]{len(templates)}[/] template(s) installed")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@template_group.command("get")
@click.argument("name")
@click.pass_context
def template_get(ctx: click.Context, name: str) -> None:
    """Show details for a template.

    Example:
        formbridge template get irs-1065-2025
    """
    verbose = ctx.obj.get("verbose", False)
    output_format = ctx.obj.get("format", "json")

    try:
        manager = TemplateManager()
        template = manager.get(name)

        if output_format == "json":
            console.print_json(template.manifest.model_dump_json(indent=2))
        else:
            console.print(Panel(
                Text.from_markup(
                    f"[bold]{template.manifest.display_name}[/]\n"
                    f"Name: [cyan]{template.manifest.name}[/]\n"
                    f"Version: {template.manifest.version}\n"
                    f"Category: {template.manifest.category or 'N/A'}\n"
                    f"Fields: {template.manifest.fields_count}\n"
                    f"Pages: {template.manifest.pages}\n"
                    f"Has instructions: {'Yes' if template.manifest.has_instructions else 'No'}\n"
                    f"Created: {template.manifest.created_at}\n"
                    f"FormBridge version: {template.manifest.formbridge_version}"
                ),
                title="Template Details",
                border_style="blue",
            ))

            if template.manifest.tags:
                console.print(f"\n[bold]Tags:[/] {', '.join(template.manifest.tags)}")

            if template.manifest.schedules:
                console.print(f"[bold]Schedules:[/] {', '.join(template.manifest.schedules)}")

            # Show field summary
            if template.schema.fields:
                console.print(f"\n[bold]Fields (first 10):[/]")
                for field in template.schema.fields[:10]:
                    label = field.label or field.id
                    line_ref = f" (Line {field.line_ref})" if field.line_ref else ""
                    console.print(f"  • {label}{line_ref} [{field.type.value}]")

                if len(template.schema.fields) > 10:
                    console.print(f"  [dim]... and {len(template.schema.fields) - 10} more[/]")

    except TemplateNotFoundError:
        console.print(f"[red]Error:[/] Template '{name}' not found")
        sys.exit(1)
    except TemplateError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


@template_group.command("delete")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def template_delete(ctx: click.Context, name: str, force: bool) -> None:
    """Delete a template.

    Example:
        formbridge template delete my-form
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        manager = TemplateManager()

        # Check if template exists
        if not manager.exists(name):
            console.print(f"[red]Error:[/] Template '{name}' not found")
            sys.exit(1)

        # Confirm deletion
        if not force:
            if not click.confirm(f"Delete template '{name}'?"):
                console.print("[yellow]Cancelled[/]")
                return

        manager.delete(name)
        console.print(f"[green]✓[/] Template '{name}' deleted")

    except TemplateError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


@template_group.command("install")
@click.argument("name")
@click.option("--registry", help="Custom registry URL")
@click.pass_context
def template_install(ctx: click.Context, name: str, registry: str | None) -> None:
    """Install a template from the registry.

    Downloads a template from the GitHub registry and installs it locally.

    Example:
        formbridge template install irs-1065-2025
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        manager = TemplateManager()

        with console.status(f"[bold green]Installing template '{name}'..."):
            template = manager.install(name, registry_url=registry, verbose=verbose)

        console.print(f"[green]✓[/] Template '{name}' installed successfully")
        console.print(f"  Display name: {template.manifest.display_name}")
        console.print(f"  Fields: {template.manifest.fields_count}")
        console.print(f"  Pages: {template.manifest.pages}")

    except TemplateAlreadyExistsError:
        console.print(f"[yellow]Template '{name}' is already installed[/]")
        console.print("Delete it first to reinstall: formbridge template delete " + name)
    except TemplateNotFoundError:
        console.print(f"[red]Error:[/] Template '{name}' not found in registry")
        sys.exit(1)
    except TemplateError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


# =============================================================================
# Data Template Command
# =============================================================================


@main.command("data-template")
@click.argument("template_name")
@click.option("--output", "-o", type=click.Path(), help="Output file for data template JSON")
@click.pass_context
def data_template_command(
    ctx: click.Context,
    template_name: str,
    output: str | None,
) -> None:
    """Generate a blank data template for a form.

    Creates a JSON file with all expected keys based on the form schema.
    Fill in the values and use with 'formbridge fill'.

    Example:
        formbridge data-template irs-1065-2025 --output my-data.json
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        manager = TemplateManager()
        template = manager.get(template_name)

        # Generate data template
        data_template = generate_data_template(template)

        # Remove descriptions for clean output (they're metadata)
        output_data = {k: v for k, v in data_template.items() if not k.startswith("_")}

        if output:
            output_path = Path(output)
            output_path.write_text(json.dumps(output_data, indent=2))
            console.print(f"[green]✓[/] Data template written to {output}")
            console.print(f"\nFill in the values and use:")
            console.print(f"  formbridge fill {template_name} --data {output} --output filled.pdf")
        else:
            console.print_json(json.dumps(output_data, indent=2))

        # Show field descriptions if available
        if "_field_descriptions" in data_template:
            console.print("\n[bold]Field descriptions:[/]")
            for key, desc in list(data_template["_field_descriptions"].items())[:10]:
                console.print(f"  [cyan]{key}[/]: {desc}")
            if len(data_template["_field_descriptions"]) > 10:
                console.print(f"  [dim]... and {len(data_template['_field_descriptions']) - 10} more[/]")

    except TemplateNotFoundError:
        console.print(f"[red]Error:[/] Template '{template_name}' not found")
        sys.exit(1)
    except TemplateError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


# =============================================================================
# Serve Command
# =============================================================================


@main.command("serve")
@click.option("--port", "-p", default=3000, help="Server port (HTTP mode)")
@click.option("--stdio", is_flag=True, help="Run in stdio mode for MCP")
@click.pass_context
def serve(ctx: click.Context, port: int, stdio: bool) -> None:
    """Start the MCP server.

    Enables FormBridge as an MCP server for AI agents like Claude Desktop
    or any MCP-compatible client.

    Modes:
      --stdio    Standard input/output mode (for Claude Desktop, etc.)
      --port N   HTTP/SSE mode on port N

    Examples:
      formbridge serve --stdio
      formbridge serve --port 3000
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        from formbridge.mcp_server import run_stdio, run_http
    except ImportError as e:
        console.print("[red]Error:[/] MCP server dependencies not installed")
        console.print("Install with: pip install formbridge[mcp]")
        console.print(f"\nDetails: {e}")
        sys.exit(1)

    if stdio:
        console.print("[dim]Starting MCP server in stdio mode...[/]", err=True)
        import asyncio
        asyncio.run(run_stdio())
    else:
        console.print(f"[bold green]Starting MCP server on port {port}[/]")
        console.print(f"  SSE endpoint: http://localhost:{port}/sse")
        console.print(f"  Messages: http://localhost:{port}/messages")
        console.print()
        console.print("[dim]Press Ctrl+C to stop[/]")
        run_http(port=port)


if __name__ == "__main__":
    main()
