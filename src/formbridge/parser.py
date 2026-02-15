"""Parser module for extracting instructions from PDF documents.

This module extracts per-field guidance from instruction documents (like IRS
instruction booklets) and maps them to field identifiers from a FormSchema.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pdfplumber

from formbridge.models import CalculationRule, FieldInstruction, FormSchema, InstructionMap

if TYPE_CHECKING:
    from formbridge.llm import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Base error for parser operations."""
    pass


class InstructionExtractionError(ParserError):
    """Error extracting instructions from PDF."""
    pass


class LLMMappingError(ParserError):
    """Error during LLM field mapping."""
    pass


@dataclass
class TextSection:
    """A section of extracted text with metadata."""
    heading: str | None
    content: str
    page_number: int
    level: int  # Heading level (0 = no heading, 1 = top level, etc.)


@dataclass
class ParsedInstruction:
    """Parsed instruction data before mapping to fields."""
    line_ref: str | None
    label: str | None
    instruction: str
    examples: list[str] | None
    constraints: list[str] | None
    format: str | None
    source_page: int | None
    source_heading: str | None


@dataclass
class ExtractedCalculationRule:
    """Extracted calculation rule before mapping to field IDs."""
    target_line_ref: str
    formula_description: str
    source_page: int | None


class InstructionExtractor:
    """Extract structured text from instruction PDFs."""

    # Patterns for detecting line references in text
    LINE_REF_PATTERNS = [
        r"[Ll]ine\s+(\d+[a-zA-Z]?)",  # "Line 16", "Line 1a"
        r"^[\s]*([A-Z])\.\s",  # "A. Name of partnership"
        r"^\s*\(\s*([a-zA-Z])\s*\)",  # "(A) Name" or "(a) Description"
    ]

    # Patterns for calculation rules
    CALCULATION_PATTERNS = [
        r"[Ll]ine\s+(\d+[a-zA-Z]?)\s*=\s*.*[Ll]ine\s+(\d+[a-zA-Z]?)",
        r"[Ss]ubtract\s+.*[Ll]ine\s+(\d+[a-zA-Z]?)\s+from\s+.*[Ll]ine\s+(\d+[a-zA-Z]?)",
        r"[Aa]dd\s+.*[Ll]ines?\s+(\d+[a-zA-Z]?)",
        r"[Ss]um\s+of\s+.*[Ll]ines?\s+(\d+[a-zA-Z]?)",
    ]

    def __init__(self, pdf_path: str | Path) -> None:
        """Initialize extractor with PDF path."""
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise ParserError(f"PDF file not found: {self.pdf_path}")

    def extract_sections(self) -> list[TextSection]:
        """Extract text sections organized by headings.

        Returns:
            List of TextSection objects with heading, content, and page info
        """
        sections: list[TextSection] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_sections = self._extract_page_sections(page, page_num)
                sections.extend(page_sections)

        return sections

    def _extract_page_sections(self, page, page_num: int) -> list[TextSection]:
        """Extract sections from a single page."""
        sections: list[TextSection] = []

        # Get all text from page
        text = page.extract_text()
        if not text:
            return sections

        # Split into lines and identify headings
        lines = text.split("\n")
        current_heading: str | None = None
        current_content: list[str] = []
        current_level = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a heading
            heading_level = self._detect_heading_level(line)

            if heading_level > 0:
                # Save previous section if exists
                if current_content:
                    sections.append(TextSection(
                        heading=current_heading,
                        content="\n".join(current_content).strip(),
                        page_number=page_num,
                        level=current_level,
                    ))

                # Start new section
                current_heading = line
                current_level = heading_level
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            sections.append(TextSection(
                heading=current_heading,
                content="\n".join(current_content).strip(),
                page_number=page_num,
                level=current_level,
            ))

        return sections

    def _detect_heading_level(self, line: str) -> int:
        """Detect if a line is a heading and return its level.

        Returns:
            0 if not a heading, 1-4 for heading levels
        """
        # Check for numeric headings (1., 1.1., 1.1.1., etc.)
        match = re.match(r"^(\d+(?:\.\d+)*)\.\s+", line)
        if match:
            # Count depth by number of dots
            level = match.group(1).count(".") + 1
            return min(level, 4)

        # Check for "Line X" pattern (common in IRS forms)
        if re.match(r"^[Ll]ine\s+\d+[a-zA-Z]?\b", line):
            return 2

        # Check for uppercase short lines (likely headings)
        if len(line) < 60 and line.isupper():
            return 2

        # Check for bold indicator (if available in text extraction)
        # This is a heuristic - might need adjustment
        if re.match(r"^(?:Specific|General|Note|Important|See also)", line):
            return 3

        return 0

    def extract_text_by_line_reference(self) -> dict[str, str]:
        """Extract text organized by line reference.

        Returns:
            Dict mapping line references (e.g., "1", "A", "16a") to text
        """
        sections = self.extract_sections()
        line_sections: dict[str, str] = {}

        for section in sections:
            if section.heading:
                # Try to extract line reference from heading
                line_ref = self._extract_line_ref_from_text(section.heading)
                if line_ref:
                    content = f"{section.heading}\n{section.content}"
                    line_sections[line_ref] = content

        return line_sections

    def _extract_line_ref_from_text(self, text: str) -> str | None:
        """Extract line reference from text."""
        # Pattern: "Line 16" or "Line 16a"
        match = re.search(r"[Ll]ine\s+(\d+[a-zA-Z]?)", text)
        if match:
            return match.group(1)

        # Pattern: "A. " at start
        match = re.match(r"^([A-Z])\.\s", text)
        if match:
            return match.group(1)

        # Pattern: "(A) " or "(a) "
        match = re.match(r"^\(\s*([a-zA-Z])\s*\)", text)
        if match:
            return match.group(1).upper()

        return None


class InstructionLLMMapper:
    """Use LLM to map extracted text to form fields."""

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize with LLM provider."""
        self.provider = provider

    def map_instructions_to_fields(
        self,
        sections: list[TextSection],
        form_schema: FormSchema,
    ) -> dict[str, FieldInstruction]:
        """Map extracted sections to form fields using LLM.

        Args:
            sections: Extracted text sections
            form_schema: Form schema with field definitions

        Returns:
            Mapping of field IDs to field instructions
        """
        # Build prompt with form fields and sections
        prompt = self._build_mapping_prompt(sections, form_schema)

        # Define output schema
        output_schema = {
            "type": "object",
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_id": {"type": "string"},
                            "line_ref": {"type": "string"},
                            "label": {"type": "string"},
                            "instruction": {"type": "string"},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "constraints": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "format": {"type": "string"},
                            "source_page": {"type": "integer"},
                        },
                        "required": ["field_id", "instruction"],
                    },
                },
            },
            "required": ["mappings"],
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a form instruction parser. Your task is to map "
                    "instruction text to form fields. Extract the specific "
                    "instruction for each field, any examples provided, "
                    "constraints, and format requirements. Be precise and "
                    "only include information explicitly stated in the instructions."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.provider.complete(messages, schema=output_schema)
        except Exception as e:
            raise LLMMappingError(f"LLM mapping failed: {e}") from e

        # Parse response
        result = response.get("content", {})
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                raise LLMMappingError(f"Invalid JSON response: {result}") from e

        mappings = result.get("mappings", [])
        field_instructions: dict[str, FieldInstruction] = {}

        for mapping in mappings:
            field_id = mapping.get("field_id")
            if not field_id:
                continue

            # Validate field exists in schema
            field_exists = any(f.id == field_id for f in form_schema.fields)
            if not field_exists:
                logger.warning(f"LLM mapped to unknown field: {field_id}")
                continue

            field_instructions[field_id] = FieldInstruction(
                line_ref=mapping.get("line_ref"),
                label=mapping.get("label"),
                instruction=mapping.get("instruction", ""),
                examples=mapping.get("examples"),
                constraints=mapping.get("constraints"),
                format=mapping.get("format"),
                source_page=mapping.get("source_page"),
            )

        return field_instructions

    def _build_mapping_prompt(
        self,
        sections: list[TextSection],
        form_schema: FormSchema,
    ) -> str:
        """Build the LLM prompt for field mapping."""
        # Build field list
        field_list = []
        for field in form_schema.fields:
            field_info = f"- {field.id}"
            if field.line_ref:
                field_info += f" (Line {field.line_ref})"
            if field.label:
                field_info += f": {field.label}"
            field_list.append(field_info)

        # Build sections text
        sections_text = []
        for section in sections:
            if section.heading:
                sections_text.append(f"\n## {section.heading} (Page {section.page_number})")
            else:
                sections_text.append(f"\n## Page {section.page_number}")
            sections_text.append(section.content)

        prompt = f"""Given the following form fields and instruction text, map each field to its specific instructions.

FORM FIELDS:
{chr(10).join(field_list)}

INSTRUCTION SECTIONS:
{chr(10).join(sections_text)}

For each field that has instructions in the text above, provide:
1. field_id: The field ID from the list above
2. line_ref: The line reference (e.g., "1", "A", "16a") if mentioned
3. label: The field label/name
4. instruction: The complete instruction text for this field
5. examples: Any example values shown (as a list)
6. constraints: Any constraints or requirements (as a list)
7. format: Expected format (e.g., "XX-XXXXXXX" for EIN)
8. source_page: The page number where this instruction appears

Only include fields that have explicit instructions in the text. Do not guess or infer instructions for fields not mentioned."""

        return prompt

    def extract_calculation_rules(
        self,
        sections: list[TextSection],
        form_schema: FormSchema,
    ) -> list[CalculationRule]:
        """Extract calculation rules from instruction sections.

        Args:
            sections: Extracted text sections
            form_schema: Form schema for field reference mapping

        Returns:
            List of calculation rules
        """
        # Build prompt for calculation extraction
        prompt = self._build_calculation_prompt(sections, form_schema)

        output_schema = {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_field_id": {"type": "string"},
                            "target_line_ref": {"type": "string"},
                            "formula": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["target_line_ref", "formula"],
                    },
                },
            },
            "required": ["rules"],
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a calculation rule extractor. Extract mathematical "
                    "formulas from form instructions. Express formulas using "
                    "field references (e.g., 'line_6 - line_7' or 'sum(line_1, line_2)'). "
                    "Be precise and only extract rules explicitly stated in the instructions."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.provider.complete(messages, schema=output_schema)
        except Exception as e:
            raise LLMMappingError(f"LLM calculation extraction failed: {e}") from e

        result = response.get("content", {})
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                raise LLMMappingError(f"Invalid JSON response: {result}") from e

        rules_data = result.get("rules", [])
        rules: list[CalculationRule] = []

        # Build line_ref to field_id mapping
        line_to_field: dict[str, str] = {}
        for field in form_schema.fields:
            if field.line_ref:
                line_to_field[field.line_ref] = field.id

        for rule_data in rules_data:
            target_line = rule_data.get("target_line_ref", "")
            target_field = rule_data.get("target_field_id") or line_to_field.get(target_line)

            if not target_field:
                logger.warning(f"Could not map calculation target: {target_line}")
                continue

            rules.append(CalculationRule(
                target=target_field,
                line_ref=target_line,
                formula=rule_data.get("formula", ""),
                description=rule_data.get("description"),
            ))

        return rules

    def _build_calculation_prompt(
        self,
        sections: list[TextSection],
        form_schema: FormSchema,
    ) -> str:
        """Build the prompt for calculation rule extraction."""
        # Build line reference mapping
        line_refs: dict[str, str] = {}
        for field in form_schema.fields:
            if field.line_ref:
                line_refs[field.line_ref] = field.id

        sections_text = []
        for section in sections:
            if section.heading:
                sections_text.append(f"\n{section.heading} (Page {section.page_number})")
            sections_text.append(section.content)

        prompt = f"""Extract calculation rules from the following form instructions.

LINE REFERENCE TO FIELD ID MAPPING:
{chr(10).join(f"- Line {line} -> {field_id}" for line, field_id in sorted(line_refs.items()))}

INSTRUCTION TEXT:
{chr(10).join(sections_text)}

Extract any calculation rules described in the text. For each rule, provide:
1. target_field_id: The field ID (from mapping above) where the result goes
2. target_line_ref: The line reference (e.g., "8", "22")
3. formula: Mathematical formula using field references (e.g., "field_040 - field_041" or "sum(field_001, field_002)")
4. description: Human-readable description (e.g., "Subtract line 7 from line 6")

Examples of calculation descriptions to look for:
- "Line 8 equals Line 6 minus Line 7"
- "Add lines 1 through 5"
- "Subtract line 21 from line 20"
- "Enter the total of all amounts"

Only extract rules that are explicitly stated in the instructions."""

        return prompt


class InstructionCache:
    """Cache for parsed instruction results.

    Instructions don't change within a tax year, so we cache parsed results
    keyed by file hash to avoid re-parsing.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files (default: ~/.cache/formbridge)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "formbridge"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, pdf_path: str | Path, form_id: str) -> str:
        """Generate cache key from file content and form ID."""
        # Hash file content + form_id
        content_hash = hashlib.sha256()
        content_hash.update(form_id.encode())

        # Add file size and mtime to hash (faster than hashing entire file)
        path = Path(pdf_path)
        if path.exists():
            stat = path.stat()
            content_hash.update(f"{stat.st_size}:{stat.st_mtime}".encode())
        else:
            # If file doesn't exist, use path as part of key
            content_hash.update(str(path).encode())

        return content_hash.hexdigest()

    def get_cache_path(self, pdf_path: str | Path, form_id: str) -> Path:
        """Get path to cache file."""
        cache_key = self._get_cache_key(pdf_path, form_id)
        return self.cache_dir / f"{cache_key}.json"

    def get(self, pdf_path: str | Path, form_id: str) -> InstructionMap | None:
        """Get cached instruction map if available.

        Args:
            pdf_path: Path to instruction PDF
            form_id: Form identifier

        Returns:
            Cached InstructionMap or None if not cached
        """
        cache_path = self.get_cache_path(pdf_path, form_id)

        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            return InstructionMap.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def set(self, pdf_path: str | Path, form_id: str, instruction_map: InstructionMap) -> None:
        """Cache instruction map.

        Args:
            pdf_path: Path to instruction PDF
            form_id: Form identifier
            instruction_map: Instruction map to cache
        """
        cache_path = self.get_cache_path(pdf_path, form_id)

        try:
            cache_path.write_text(instruction_map.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def clear(self) -> None:
        """Clear all cached instructions."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")


class Parser:
    """Main parser class that coordinates instruction extraction."""

    def __init__(
        self,
        instructions_path: str | Path,
        schema: FormSchema | None = None,
        llm_config: "LLMConfig" | None = None,
        use_cache: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize parser.

        Args:
            instructions_path: Path to instruction PDF
            schema: Form schema for field mapping
            llm_config: LLM configuration for field mapping
            use_cache: Whether to use caching
            verbose: Enable verbose logging
        """
        self.instructions_path = Path(instructions_path)
        self.schema = schema
        self.llm_config = llm_config
        self.use_cache = use_cache
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        self.extractor = InstructionExtractor(self.instructions_path)
        self.cache = InstructionCache() if use_cache else None

        # Initialize provider if config provided
        self._provider: LLMProvider | None = None

    @property
    def provider(self) -> LLMProvider | None:
        """Get or create LLM provider."""
        if self._provider is None and self.llm_config:
            from formbridge.llm import create_provider

            self._provider = create_provider(
                provider=self.llm_config.provider,
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
            )
        return self._provider

    def parse(self) -> InstructionMap:
        """Parse instructions and map to fields.

        Returns:
            InstructionMap with parsed instructions
        """
        logger.info(f"Parsing instructions: {self.instructions_path}")

        if self.schema is None:
            raise ParserError("Form schema is required for parsing. Use --fields to provide a schema.")

        form_id = self.schema.form_id

        # Check cache first
        if self.use_cache and self.cache:
            cached = self.cache.get(self.instructions_path, form_id)
            if cached:
                logger.info("Using cached instruction map")
                return cached

        # Extract text sections
        sections = self.extractor.extract_sections()
        logger.info(f"Extracted {len(sections)} text sections")

        # Create base instruction map
        instruction_map = InstructionMap(
            form_id=form_id,
            field_instructions={},
            calculation_rules=[],
        )

        # If no provider, we can't do LLM mapping
        if not self.provider:
            logger.warning("No LLM provider configured, skipping field mapping")
            return instruction_map

        # Use LLM to map instructions to fields
        mapper = InstructionLLMMapper(self.provider)

        # Map field instructions
        field_instructions = mapper.map_instructions_to_fields(sections, self.schema)
        instruction_map.field_instructions = field_instructions
        logger.info(f"Mapped instructions for {len(field_instructions)} fields")

        # Extract calculation rules
        calculation_rules = mapper.extract_calculation_rules(sections, self.schema)
        instruction_map.calculation_rules = calculation_rules
        logger.info(f"Extracted {len(calculation_rules)} calculation rules")

        # Cache result
        if self.use_cache and self.cache:
            self.cache.set(self.instructions_path, form_id, instruction_map)

        return instruction_map

    def to_json(self, instruction_map: InstructionMap | None = None) -> str:
        """Export instruction map as JSON string.

        Args:
            instruction_map: Instruction map to export (if None, parses first)

        Returns:
            JSON string
        """
        if instruction_map is None:
            instruction_map = self.parse()

        return instruction_map.model_dump_json(indent=2)


def parse_instructions(
    instructions_path: str | Path,
    form_schema: FormSchema | None = None,
    provider: LLMProvider | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> InstructionMap:
    """Convenience function to parse instructions.

    Args:
        instructions_path: Path to instruction PDF
        form_schema: Optional form schema for field mapping
        provider: LLM provider for mapping
        use_cache: Whether to use caching
        verbose: Enable verbose logging

    Returns:
        InstructionMap with parsed instructions
    """
    parser = Parser(
        instructions_path=instructions_path,
        schema=form_schema,
        use_cache=use_cache,
        verbose=verbose,
    )
    return parser.parse()
