"""Pydantic models for FormBridge data structures.

These models define the core data structures used throughout FormBridge.
They match the spec exactly to ensure compatibility with all modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Type of form field."""

    TEXT = "text"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DATE = "date"
    NUMBER = "number"
    SIGNATURE = "signature"


class FieldPosition(BaseModel):
    """Position of a field on a PDF page.

    Coordinates are in PDF points (1/72 inch) with origin at bottom-left.
    """

    x: float = Field(description="X coordinate (left edge)")
    y: float = Field(description="Y coordinate (bottom edge)")
    w: float = Field(description="Width of field")
    h: float = Field(description="Height of field")


class FormField(BaseModel):
    """A single fillable field in a PDF form."""

    id: str = Field(description="Unique field identifier (e.g., 'field_001')")
    label: str | None = Field(default=None, description="Human-readable field label")
    page: int = Field(description="Page number (1-indexed)")
    type: FieldType = Field(description="Field type")
    position: FieldPosition | None = Field(default=None, description="Field position on page")
    max_length: int | None = Field(default=None, description="Maximum character length for text fields")
    required: bool = Field(default=False, description="Whether field is required")
    line_ref: str | None = Field(default=None, description="Line reference (e.g., 'A', '1', '16a')")
    constraints: dict[str, Any] | None = Field(default=None, description="Additional constraints")
    # For radio buttons
    options: list[str] | None = Field(default=None, description="Options for radio/select fields")
    # For checkboxes
    checked_value: str | None = Field(default=None, description="Value when checkbox is checked (usually 'Yes' or 'On')")
    # Label provenance (ADR 001)
    label_source: str | None = Field(
        default=None,
        description="How the label was derived: 'acroform', 'vision', 'ocr', or None",
    )
    label_confidence: float | None = Field(
        default=None,
        description="Confidence score for the label (0.0-1.0). Set by vision refinement.",
    )


class FormSchema(BaseModel):
    """Schema describing a PDF form's structure.

    This is the primary output of the Scanner module.
    """

    form_id: str = Field(description="Unique identifier for this form (e.g., 'irs-1065-2025')")
    pages: int = Field(description="Total number of pages")
    fields: list[FormField] = Field(default_factory=list, description="All fillable fields")
    source_file: str | None = Field(default=None, description="Original PDF filename")
    created_at: str | None = Field(default=None, description="ISO timestamp when schema was created")
    formbridge_version: str = Field(default="0.1.0", description="FormBridge version used to create schema")


class FieldInstruction(BaseModel):
    """Parsed instruction for a single field."""

    line_ref: str | None = Field(default=None, description="Line reference (e.g., 'A', '1')")
    label: str | None = Field(default=None, description="Field label from instructions")
    instruction: str | None = Field(default=None, description="Full instruction text")
    examples: list[str] | None = Field(default=None, description="Example values")
    constraints: list[str] | None = Field(default=None, description="Constraint descriptions")
    format: str | None = Field(default=None, description="Expected format (e.g., 'XX-XXXXXXX')")
    source_page: int | None = Field(default=None, description="Page number in instruction document")


class CalculationRule(BaseModel):
    """A calculation rule for a derived field."""

    target: str = Field(description="Target field ID")
    line_ref: str | None = Field(default=None, description="Line reference")
    formula: str = Field(description="Formula (e.g., 'field_040 - field_041')")
    description: str | None = Field(default=None, description="Human-readable description")


class InstructionMap(BaseModel):
    """Mapping of fields to their instructions.

    This is the primary output of the Parser module.
    """

    form_id: str = Field(description="Form identifier")
    field_instructions: dict[str, FieldInstruction] = Field(
        default_factory=dict,
        description="Mapping of field IDs to instructions"
    )
    calculation_rules: list[CalculationRule] = Field(
        default_factory=list,
        description="Calculation rules for derived fields"
    )


class FieldMapping(BaseModel):
    """A single field mapping with confidence score."""

    field_id: str = Field(description="Target field ID")
    value: str | None = Field(default=None, description="Value to fill")
    confidence: float = Field(description="Confidence score (0.0 - 1.0)")
    reasoning: str | None = Field(default=None, description="Why this value was chosen")
    source_key: str | None = Field(default=None, description="Source data key used")
    calculated: bool = Field(default=False, description="Whether value was calculated")
    formula: str | None = Field(default=None, description="Formula used if calculated")


class MappingWarning(BaseModel):
    """A warning about a field mapping."""

    field_id: str | None = Field(default=None, description="Field ID (if applicable)")
    message: str = Field(description="Warning message")
    severity: str = Field(default="warning", description="Severity: 'info', 'warning', 'error'")


class FieldMappingResult(BaseModel):
    """Complete field mapping result.

    This is the primary output of the Mapper module.
    """

    mappings: list[FieldMapping] = Field(default_factory=list, description="Field mappings")
    unmapped_fields: list[str] = Field(default_factory=list, description="Fields with no mapping")
    unmapped_data: list[str] = Field(default_factory=list, description="Data keys not used")
    calculations: list[FieldMapping] = Field(default_factory=list, description="Calculated field mappings")
    warnings: list[MappingWarning] = Field(default_factory=list, description="Warnings about mappings")


class VerificationReport(BaseModel):
    """Verification report for a filled form."""

    form: str = Field(description="Form identifier or filename")
    timestamp: str = Field(description="ISO timestamp")
    overall_confidence: float = Field(description="Average confidence score")
    fields_total: int = Field(description="Total fields in form")
    fields_filled: int = Field(description="Fields that were filled")
    fields_blank: int = Field(description="Intentionally blank fields")
    fields_calculated: int = Field(description="Calculated fields")
    fields_flagged: int = Field(description="Fields flagged for review")
    flags: list[MappingWarning] = Field(default_factory=list, description="All flags")


class CalculationResult(BaseModel):
    """Result of a calculation execution."""

    field_id: str = Field(description="Target field ID for the calculated value")
    formula: str = Field(description="Formula that was executed")
    value: float | int | str | None = Field(default=None, description="Calculated result value")
    source_fields: list[str] = Field(default_factory=list, description="Field IDs used in calculation")
    error: str | None = Field(default=None, description="Error message if calculation failed")
    verified: bool = Field(default=False, description="Whether result matches LLM-mapped value")


class FieldMappingEntry(BaseModel):
    """A single field mapping entry with full context."""

    field_id: str = Field(description="Target field ID")
    field_label: str | None = Field(default=None, description="Field label from schema")
    field_type: FieldType = Field(description="Field type")
    value: str | None = Field(default=None, description="Value to fill")
    confidence: float = Field(description="Confidence score (0.0 - 1.0)", ge=0.0, le=1.0)
    reasoning: str | None = Field(default=None, description="Why this value was chosen")
    source_key: str | None = Field(default=None, description="Source data key used")
    calculated: bool = Field(default=False, description="Whether value was calculated")
    formula: str | None = Field(default=None, description="Formula used if calculated")
    instruction_hint: str | None = Field(default=None, description="Relevant instruction text")
    needs_review: bool = Field(default=False, description="Whether field needs manual review")
    review_reason: str | None = Field(default=None, description="Why field needs review")


class FillResult(BaseModel):
    """Result of a fill operation."""

    form_id: str = Field(description="Form identifier")
    output_path: str | None = Field(default=None, description="Path to filled PDF")
    mapping: FieldMappingResult | None = Field(default=None, description="Field mapping used")
    verification: VerificationReport | None = Field(default=None, description="Verification report")
    success: bool = Field(description="Whether fill succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
