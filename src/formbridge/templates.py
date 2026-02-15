"""Template system for FormBridge.

Templates bundle scanned forms + parsed instructions for reuse.
They enable scan-once, fill-many workflows and can be shared via a GitHub registry.

Template structure:
    ~/.formbridge/templates/<name>/
    ├── manifest.json          # Metadata
    ├── form.pdf               # Original blank form
    ├── schema.json            # Scanner output (FormSchema)
    ├── instructions.json      # Parser output (InstructionMap) - optional
    └── instructions-source.pdf # Original instruction doc - optional
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from formbridge.models import FormSchema, InstructionMap

def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("formbridge")
    except Exception:
        return "0.1.0"

logger = logging.getLogger(__name__)


# Default templates directory
DEFAULT_TEMPLATES_DIR = Path.home() / ".formbridge" / "templates"

# GitHub registry base URL
REGISTRY_BASE_URL = "https://raw.githubusercontent.com/nilsyai/formbridge-templates/main"


class TemplateError(Exception):
    """Base error for template operations."""
    pass


class TemplateNotFoundError(TemplateError):
    """Template not found."""
    pass


class TemplateAlreadyExistsError(TemplateError):
    """Template already exists."""
    pass


class TemplateManifest(BaseModel):
    """Metadata for a form template.

    This is stored as manifest.json in the template directory.
    """

    name: str = Field(description="Template identifier (e.g., 'irs-1065-2025')")
    display_name: str = Field(description="Human-readable name")
    version: str = Field(default="1.0.0", description="Template version")
    category: str | None = Field(default=None, description="Category (e.g., 'tax/us/irs')")
    tags: list[str] = Field(default_factory=list, description="Tags for search")
    fields_count: int = Field(description="Number of fields in the form")
    pages: int = Field(description="Number of pages")
    schedules: list[str] | None = Field(default=None, description="Related schedules")
    created_at: str = Field(description="ISO timestamp when template was created")
    formbridge_version: str = Field(
        default_factory=_get_version,
        description="FormBridge version used to create template"
    )
    has_instructions: bool = Field(
        default=False,
        description="Whether template includes parsed instructions"
    )
    source_url: str | None = Field(
        default=None,
        description="URL to template in registry (if installed from registry)"
    )


class Template:
    """A loaded template with all its components."""

    def __init__(
        self,
        path: Path,
        manifest: TemplateManifest,
        schema: FormSchema,
        instruction_map: InstructionMap | None = None,
    ) -> None:
        """Initialize a template.

        Args:
            path: Path to template directory
            manifest: Template metadata
            schema: Form schema
            instruction_map: Optional instruction map
        """
        self.path = path
        self.manifest = manifest
        self.schema = schema
        self.instruction_map = instruction_map

    @property
    def name(self) -> str:
        """Get template name."""
        return self.manifest.name

    @property
    def has_instructions(self) -> bool:
        """Check if template has parsed instructions."""
        return self.instruction_map is not None and len(self.instruction_map.field_instructions) > 0


class TemplateManager:
    """Manage form templates.

    Templates are stored in ~/.formbridge/templates/ and can be:
    - Created from scanned forms + parsed instructions
    - Installed from a GitHub registry
    - Listed, loaded, and deleted
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        """Initialize the template manager.

        Args:
            templates_dir: Directory for templates (default: ~/.formbridge/templates)
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        form_pdf: str | Path,
        name: str,
        instructions_pdf: str | Path | None = None,
        display_name: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        verbose: bool = False,
    ) -> Template:
        """Create a new template from a form PDF.

        Scans the form and optionally parses instructions, then packages
        everything into a reusable template.

        Args:
            form_pdf: Path to the blank form PDF
            name: Template name (alphanumeric, hyphens, underscores only)
            instructions_pdf: Optional path to instruction document
            display_name: Human-readable name (defaults to name)
            category: Category for organization
            tags: Tags for search
            verbose: Enable verbose output

        Returns:
            The created Template

        Raises:
            TemplateAlreadyExistsError: If template already exists
            TemplateError: If creation fails
        """
        from formbridge.scanner import Scanner
        from formbridge.parser import Parser
        from formbridge.llm import load_config

        # Validate name
        if not self._is_valid_name(name):
            raise TemplateError(
                f"Invalid template name '{name}'. Use only letters, numbers, hyphens, and underscores."
            )

        # Check if template already exists
        template_path = self.templates_dir / name
        if template_path.exists():
            raise TemplateAlreadyExistsError(
                f"Template '{name}' already exists. Delete it first or use a different name."
            )

        form_pdf = Path(form_pdf)
        if not form_pdf.exists():
            raise TemplateError(f"Form PDF not found: {form_pdf}")

        if verbose:
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(level=logging.DEBUG)

        try:
            # Step 1: Scan the form
            if verbose:
                logger.info(f"Scanning form: {form_pdf}")

            scanner = Scanner(form_pdf, verbose=verbose)
            schema = scanner.scan()

            if verbose:
                logger.info(f"Scanned {len(schema.fields)} fields from {schema.pages} pages")

            # Step 2: Parse instructions if provided
            instruction_map: InstructionMap | None = None
            has_instructions = False

            if instructions_pdf:
                instructions_pdf = Path(instructions_pdf)
                if not instructions_pdf.exists():
                    raise TemplateError(f"Instructions PDF not found: {instructions_pdf}")

                if verbose:
                    logger.info(f"Parsing instructions: {instructions_pdf}")

                llm_config = load_config()
                parser = Parser(
                    instructions_path=instructions_pdf,
                    schema=schema,
                    llm_config=llm_config,
                    verbose=verbose,
                )
                instruction_map = parser.parse()
                has_instructions = len(instruction_map.field_instructions) > 0

                if verbose:
                    logger.info(
                        f"Parsed {len(instruction_map.field_instructions)} field instructions, "
                        f"{len(instruction_map.calculation_rules)} calculation rules"
                    )

            # Step 3: Create template directory structure
            template_path.mkdir(parents=True)

            # Step 4: Create manifest
            manifest = TemplateManifest(
                name=name,
                display_name=display_name or name,
                version="1.0.0",
                category=category,
                tags=tags or [],
                fields_count=len(schema.fields),
                pages=schema.pages,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                formbridge_version=_get_version(),
                has_instructions=has_instructions,
            )

            # Step 5: Write files
            self._write_template_files(
                template_path,
                manifest,
                schema,
                instruction_map,
                form_pdf,
                instructions_pdf if instructions_pdf else None,
            )

            if verbose:
                logger.info(f"Template created at: {template_path}")

            return Template(
                path=template_path,
                manifest=manifest,
                schema=schema,
                instruction_map=instruction_map,
            )

        except Exception as e:
            # Clean up on failure
            if template_path.exists():
                shutil.rmtree(template_path)
            raise TemplateError(f"Failed to create template: {e}") from e

    def _write_template_files(
        self,
        template_path: Path,
        manifest: TemplateManifest,
        schema: FormSchema,
        instruction_map: InstructionMap | None,
        form_pdf: Path,
        instructions_pdf: Path | None,
    ) -> None:
        """Write all template files to disk."""
        # Write manifest
        manifest_path = template_path / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2))

        # Write schema
        schema_path = template_path / "schema.json"
        schema_path.write_text(schema.model_dump_json(indent=2))

        # Copy form PDF
        shutil.copy2(form_pdf, template_path / "form.pdf")

        # Write instructions if available
        if instruction_map:
            inst_path = template_path / "instructions.json"
            inst_path.write_text(instruction_map.model_dump_json(indent=2))

        # Copy instructions source PDF if available
        if instructions_pdf:
            shutil.copy2(instructions_pdf, template_path / "instructions-source.pdf")

    def _is_valid_name(self, name: str) -> bool:
        """Check if template name is valid."""
        import re
        return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))

    def list(self) -> list[TemplateManifest]:
        """List all installed templates.

        Returns:
            List of TemplateManifest objects for installed templates
        """
        manifests: list[TemplateManifest] = []

        for template_dir in self.templates_dir.iterdir():
            if not template_dir.is_dir():
                continue

            manifest_path = template_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                data = json.loads(manifest_path.read_text())
                manifest = TemplateManifest.model_validate(data)
                manifests.append(manifest)
            except Exception as e:
                logger.warning(f"Failed to load template manifest {manifest_path}: {e}")

        # Sort by name
        manifests.sort(key=lambda m: m.name)

        return manifests

    def get(self, name: str) -> Template:
        """Load a template by name.

        Args:
            name: Template name

        Returns:
            Template object with all components loaded

        Raises:
            TemplateNotFoundError: If template doesn't exist
        """
        template_path = self.templates_dir / name

        if not template_path.exists():
            raise TemplateNotFoundError(f"Template '{name}' not found")

        # Load manifest
        manifest_path = template_path / "manifest.json"
        if not manifest_path.exists():
            raise TemplateError(f"Template '{name}' is missing manifest.json")

        try:
            manifest_data = json.loads(manifest_path.read_text())
            manifest = TemplateManifest.model_validate(manifest_data)
        except Exception as e:
            raise TemplateError(f"Failed to load manifest for '{name}': {e}") from e

        # Load schema
        schema_path = template_path / "schema.json"
        if not schema_path.exists():
            raise TemplateError(f"Template '{name}' is missing schema.json")

        try:
            schema_data = json.loads(schema_path.read_text())
            schema = FormSchema.model_validate(schema_data)
        except Exception as e:
            raise TemplateError(f"Failed to load schema for '{name}': {e}") from e

        # Load instructions if available
        instruction_map: InstructionMap | None = None
        instructions_path = template_path / "instructions.json"
        if instructions_path.exists():
            try:
                inst_data = json.loads(instructions_path.read_text())
                instruction_map = InstructionMap.model_validate(inst_data)
            except Exception as e:
                logger.warning(f"Failed to load instructions for '{name}': {e}")

        return Template(
            path=template_path,
            manifest=manifest,
            schema=schema,
            instruction_map=instruction_map,
        )

    def delete(self, name: str) -> bool:
        """Delete a template.

        Args:
            name: Template name

        Returns:
            True if deleted successfully

        Raises:
            TemplateNotFoundError: If template doesn't exist
        """
        template_path = self.templates_dir / name

        if not template_path.exists():
            raise TemplateNotFoundError(f"Template '{name}' not found")

        try:
            shutil.rmtree(template_path)
            return True
        except Exception as e:
            raise TemplateError(f"Failed to delete template '{name}': {e}") from e

    def install(
        self,
        name: str,
        registry_url: str | None = None,
        verbose: bool = False,
    ) -> Template:
        """Install a template from the registry.

        Downloads the template package from GitHub and installs it locally.

        Args:
            name: Template name (e.g., 'irs-1065-2025')
            registry_url: Base URL for registry (default: official FormBridge registry)
            verbose: Enable verbose output

        Returns:
            The installed Template

        Raises:
            TemplateError: If installation fails
            TemplateAlreadyExistsError: If template already exists locally
        """
        base_url = registry_url or REGISTRY_BASE_URL

        # Check if already installed
        template_path = self.templates_dir / name
        if template_path.exists():
            raise TemplateAlreadyExistsError(
                f"Template '{name}' is already installed. Delete it first to reinstall."
            )

        if verbose:
            logger.info(f"Installing template '{name}' from {base_url}")

        try:
            # Fetch manifest
            manifest_url = f"{base_url}/templates/{name}/manifest.json"
            manifest = self._fetch_json(manifest_url)

            if verbose:
                logger.info(f"Found template: {manifest.get('display_name', name)}")

            # Fetch schema
            schema_url = f"{base_url}/templates/{name}/schema.json"
            schema_data = self._fetch_json(schema_url)

            # Create template directory
            template_path.mkdir(parents=True)

            # Download form PDF
            form_url = f"{base_url}/templates/{name}/form.pdf"
            form_path = template_path / "form.pdf"
            self._download_file(form_url, form_path)

            # Try to download instructions if available
            has_instructions = manifest.get("has_instructions", False)
            instruction_map_data: dict | None = None

            if has_instructions:
                try:
                    instructions_url = f"{base_url}/templates/{name}/instructions.json"
                    instruction_map_data = self._fetch_json(instructions_url)
                except Exception as e:
                    if verbose:
                        logger.warning(f"Could not fetch instructions: {e}")
                    has_instructions = False

            # Write files locally
            validated_manifest = TemplateManifest.model_validate(manifest)
            validated_manifest.source_url = f"{base_url}/templates/{name}"

            (template_path / "manifest.json").write_text(
                validated_manifest.model_dump_json(indent=2)
            )

            validated_schema = FormSchema.model_validate(schema_data)
            (template_path / "schema.json").write_text(
                validated_schema.model_dump_json(indent=2)
            )

            instruction_map: InstructionMap | None = None
            if instruction_map_data:
                instruction_map = InstructionMap.model_validate(instruction_map_data)
                (template_path / "instructions.json").write_text(
                    instruction_map.model_dump_json(indent=2)
                )

            if verbose:
                logger.info(f"Template '{name}' installed successfully")

            return Template(
                path=template_path,
                manifest=validated_manifest,
                schema=validated_schema,
                instruction_map=instruction_map,
            )

        except Exception as e:
            # Clean up on failure
            if template_path.exists():
                shutil.rmtree(template_path)
            raise TemplateError(f"Failed to install template '{name}': {e}") from e

    def _fetch_json(self, url: str) -> dict:
        """Fetch JSON from a URL."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise TemplateNotFoundError(f"Resource not found: {url}")
            raise TemplateError(f"HTTP error fetching {url}: {e.response.status_code}")
        except Exception as e:
            raise TemplateError(f"Failed to fetch {url}: {e}")

    def _download_file(self, url: str, path: Path) -> None:
        """Download a file from a URL."""
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
        except Exception as e:
            raise TemplateError(f"Failed to download {url}: {e}")

    def get_form_path(self, name: str) -> Path:
        """Get the path to a template's form PDF.

        Args:
            name: Template name

        Returns:
            Path to the form PDF
        """
        return self.templates_dir / name / "form.pdf"

    def exists(self, name: str) -> bool:
        """Check if a template exists.

        Args:
            name: Template name

        Returns:
            True if template exists
        """
        template_path = self.templates_dir / name
        return template_path.exists() and (template_path / "manifest.json").exists()


def generate_data_template(template: Template) -> dict[str, Any]:
    """Generate a blank data template for a form.

    Creates a JSON structure with all expected keys based on the form schema.
    Users can fill this in to provide data for the fill command.

    Args:
        template: The template to generate a data template for

    Returns:
        Dict with field keys and example/placeholder values
    """
    data_template: dict[str, Any] = {}

    for field in template.schema.fields:
        # Generate a key from the field
        key = _field_to_key(field)

        # Skip if we already have this key (dedup by key)
        if key in data_template:
            continue

        # Generate placeholder based on field type
        placeholder = _generate_placeholder(field, template)

        # Add description if available
        if template.instruction_map:
            inst = template.instruction_map.field_instructions.get(field.id)
            if inst and inst.instruction:
                # Use comment-style annotation for JSON
                # Note: JSON doesn't support comments, so we'll use a special _description field
                pass

        data_template[key] = placeholder

    # Add a metadata section with field descriptions
    descriptions: dict[str, str] = {}
    for field in template.schema.fields:
        key = _field_to_key(field)
        if key in descriptions:
            continue

        desc_parts = []
        if field.label:
            desc_parts.append(field.label)
        if field.line_ref:
            desc_parts.append(f"(Line {field.line_ref})")
        if field.type:
            desc_parts.append(f"[{field.type.value}]")

        if template.instruction_map:
            inst = template.instruction_map.field_instructions.get(field.id)
            if inst and inst.instruction:
                desc_parts.append(f": {inst.instruction[:100]}...")

        if desc_parts:
            descriptions[key] = " ".join(desc_parts)

    if descriptions:
        data_template["_field_descriptions"] = descriptions

    return data_template


def _field_to_key(field) -> str:
    """Convert a field to a data key name."""
    # Try label first, clean it up
    if field.label:
        key = field.label.lower()
        key = key.replace(" ", "_").replace("-", "_")
        # Remove special characters
        import re
        key = re.sub(r"[^a-z0-9_]", "", key)
        return key

    # Fall back to line reference
    if field.line_ref:
        return f"line_{field.line_ref.lower()}"

    # Last resort: use field ID
    return field.id


def _generate_placeholder(field, template: Template) -> Any:
    """Generate a placeholder value for a field."""
    from formbridge.models import FieldType

    # Check instructions for examples
    if template.instruction_map:
        inst = template.instruction_map.field_instructions.get(field.id)
        if inst and inst.examples and len(inst.examples) > 0:
            return inst.examples[0]

    # Generate based on type
    if field.type == FieldType.TEXT:
        if field.max_length:
            return ""
        return ""
    elif field.type == FieldType.NUMBER:
        return 0
    elif field.type == FieldType.DATE:
        return "YYYY-MM-DD"
    elif field.type == FieldType.CHECKBOX:
        return False
    elif field.type == FieldType.RADIO:
        if field.options:
            return field.options[0]
        return ""
    else:
        return ""


# Convenience functions

def create_template(
    form_pdf: str | Path,
    name: str,
    instructions_pdf: str | Path | None = None,
    **kwargs,
) -> Template:
    """Convenience function to create a template."""
    manager = TemplateManager()
    return manager.create(form_pdf, name, instructions_pdf=instructions_pdf, **kwargs)


def get_template(name: str) -> Template:
    """Convenience function to load a template."""
    manager = TemplateManager()
    return manager.get(name)


def list_templates() -> list[TemplateManifest]:
    """Convenience function to list templates."""
    manager = TemplateManager()
    return manager.list()
