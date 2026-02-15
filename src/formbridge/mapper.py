"""Mapper module for mapping user data to form fields.

This module uses LLM to intelligently map user-provided data to form fields,
guided by parsed instructions. It handles:
- Data-to-field matching with confidence scoring
- Programmatic calculation execution (NOT LLM-based)
- Cross-checking LLM-mapped calculated fields against independent calculation
- Flagging fields with low confidence for human review
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from formbridge.models import (
    CalculationResult,
    CalculationRule,
    FieldInstruction,
    FieldMapping,
    FieldMappingEntry,
    FieldMappingResult,
    FieldType,
    FormField,
    FormSchema,
    InstructionMap,
    MappingWarning,
)

if TYPE_CHECKING:
    from formbridge.llm import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class MapperError(Exception):
    """Base error for mapper operations."""
    pass


class DataLoadError(MapperError):
    """Error loading user data."""
    pass


class CalculationError(MapperError):
    """Error executing calculation."""
    pass


# Confidence thresholds
CONFIDENCE_AUTO_FILL = 0.95  # >= this: auto-fill without review
CONFIDENCE_REVIEW = 0.80    # >= this and < 0.95: fill but flag for review
# < 0.80: leave blank, require human input


@dataclass
class MappingContext:
    """Context for field mapping."""
    field: FormField
    instruction: FieldInstruction | None
    user_data_value: Any | None
    user_data_key: str | None


class CalculationExecutor:
    """Execute calculation formulas programmatically.

    This is pure Python math - NO LLM involved for calculations.
    This ensures accuracy and consistency for derived values.
    """

    # Safe functions available in formulas
    SAFE_FUNCTIONS = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": lambda *args: sum(args),
        "round": round,
        "int": int,
        "float": float,
    }

    def __init__(self, field_values: dict[str, float | int | None]) -> None:
        """Initialize with current field values.

        Args:
            field_values: Mapping of field_id to current value (for formula evaluation)
        """
        self.field_values = field_values

    def execute(self, formula: str) -> CalculationResult:
        """Execute a calculation formula.

        Args:
            formula: Formula string (e.g., "field_040 - field_041")

        Returns:
            CalculationResult with value or error
        """
        # Extract field references from formula
        source_fields = self._extract_field_refs(formula)

        # Check all source fields have values
        missing_fields = [
            f for f in source_fields
            if f not in self.field_values or self.field_values[f] is None
        ]

        if missing_fields:
            return CalculationResult(
                field_id="",
                formula=formula,
                value=None,
                source_fields=source_fields,
                error=f"Missing values for fields: {', '.join(missing_fields)}",
            )

        # Build evaluation context
        eval_context = self._build_eval_context(source_fields)

        try:
            # Sanitize and evaluate
            sanitized_formula = self._sanitize_formula(formula)
            result = eval(sanitized_formula, {"__builtins__": {}}, eval_context)

            # Convert result to appropriate type
            if isinstance(result, float):
                # Round to 2 decimal places for currency
                result = round(result, 2)
            elif isinstance(result, int):
                pass
            else:
                result = str(result)

            return CalculationResult(
                field_id="",
                formula=formula,
                value=result,
                source_fields=source_fields,
                error=None,
            )

        except Exception as e:
            return CalculationResult(
                field_id="",
                formula=formula,
                value=None,
                source_fields=source_fields,
                error=f"Calculation error: {e}",
            )

    def _extract_field_refs(self, formula: str) -> list[str]:
        """Extract field IDs referenced in formula.

        Supports patterns:
        - field_001, field_042 (standard field IDs)
        - line_1, line_16a (line references)
        """
        # Match field_XXX pattern
        field_pattern = r"field_\d+"
        # Match line_XXX pattern
        line_pattern = r"line_\d+[a-zA-Z]?"

        matches = re.findall(f"{field_pattern}|{line_pattern}", formula)
        return list(set(matches))

    def _sanitize_formula(self, formula: str) -> str:
        """Sanitize formula for safe evaluation.

        Replaces field references with variable names and ensures
        only safe operations are allowed.
        """
        # Already sanitized by extracting field refs - just ensure no dangerous operations
        dangerous = ["import", "exec", "eval", "__", "open", "file", "os", "sys"]
        for word in dangerous:
            if word in formula.lower():
                raise CalculationError(f"Dangerous operation in formula: {word}")

        return formula

    def _build_eval_context(self, source_fields: list[str]) -> dict[str, Any]:
        """Build evaluation context with field values and safe functions."""
        context = dict(self.SAFE_FUNCTIONS)

        for field_id in source_fields:
            # Convert field_XXX to valid Python identifier
            var_name = field_id
            value = self.field_values.get(field_id)
            if value is not None:
                # Ensure numeric
                if isinstance(value, str):
                    try:
                        value = float(value.replace(",", "").replace("$", ""))
                    except ValueError:
                        value = 0
                context[var_name] = value

        return context


class DataToFieldMapper:
    """Use LLM to map user data to form fields."""

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize with LLM provider."""
        self.provider = provider

    def map_data_to_fields(
        self,
        user_data: dict[str, Any],
        form_schema: FormSchema,
        instruction_map: InstructionMap | None = None,
    ) -> list[FieldMapping]:
        """Map user data keys to form fields using LLM.

        Args:
            user_data: User-provided data dictionary
            form_schema: Form schema with field definitions
            instruction_map: Optional instruction map for guidance

        Returns:
            List of FieldMapping objects with confidence scores
        """
        prompt = self._build_mapping_prompt(user_data, form_schema, instruction_map)

        output_schema = {
            "type": "object",
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_id": {"type": "string"},
                            "value": {"type": ["string", "null"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reasoning": {"type": "string"},
                            "source_key": {"type": ["string", "null"]},
                        },
                        "required": ["field_id", "value", "confidence"],
                    },
                },
                "unmapped_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "unmapped_data": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["mappings"],
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a form-filling assistant. Your task is to map user data "
                    "to the correct form fields. For each mapping:\n"
                    "1. Match user data keys to the most appropriate field\n"
                    "2. Follow any instructions exactly\n"
                    "3. Apply format requirements (dates, EIN format, etc.)\n"
                    "4. Assign a confidence score (0.0 - 1.0)\n"
                    "5. Provide reasoning for each mapping\n\n"
                    "Confidence scoring guidelines:\n"
                    "- 0.95-1.0: Exact match, clear instruction, high confidence\n"
                    "- 0.80-0.94: Likely match, some ambiguity\n"
                    "- 0.60-0.79: Uncertain, needs human review\n"
                    "- Below 0.60: No suitable data found, leave blank\n\n"
                    "Return a JSON object with mappings array."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.provider.complete(messages, schema=output_schema)
        except Exception as e:
            raise MapperError(f"LLM mapping failed: {e}") from e

        result = response.get("content", {})
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                raise MapperError(f"Invalid JSON response: {result}") from e

        # Parse mappings
        mappings: list[FieldMapping] = []
        for mapping_data in result.get("mappings", []):
            field_id = mapping_data.get("field_id")
            if not field_id:
                continue

            # Validate field exists
            field_exists = any(f.id == field_id for f in form_schema.fields)
            if not field_exists:
                logger.warning(f"LLM mapped to unknown field: {field_id}")
                continue

            # Get value - handle various types
            value = mapping_data.get("value")
            if value is not None:
                value = str(value)

            mappings.append(FieldMapping(
                field_id=field_id,
                value=value,
                confidence=float(mapping_data.get("confidence", 0.5)),
                reasoning=mapping_data.get("reasoning"),
                source_key=mapping_data.get("source_key"),
            ))

        return mappings

    def _build_mapping_prompt(
        self,
        user_data: dict[str, Any],
        form_schema: FormSchema,
        instruction_map: InstructionMap | None,
    ) -> str:
        """Build the LLM prompt for data-to-field mapping."""
        # Build field list with instructions
        field_list = []
        for field in form_schema.fields:
            field_info = f"- {field.id} (Page {field.page}, Type: {field.type.value})"
            if field.label:
                field_info += f": {field.label}"
            if field.line_ref:
                field_info += f" [Line {field.line_ref}]"

            # Add instruction if available
            if instruction_map and field.id in instruction_map.field_instructions:
                inst = instruction_map.field_instructions[field.id]
                if inst.instruction:
                    field_info += f"\n  Instruction: {inst.instruction}"
                if inst.format:
                    field_info += f"\n  Format: {inst.format}"
                if inst.examples:
                    field_info += f"\n  Examples: {', '.join(inst.examples)}"

            field_list.append(field_info)

        # Build calculation rules
        calc_rules = []
        if instruction_map and instruction_map.calculation_rules:
            for rule in instruction_map.calculation_rules:
                calc_rules.append(
                    f"- {rule.target} (Line {rule.line_ref or '?'}): {rule.formula}"
                    f"{f' - {rule.description}' if rule.description else ''}"
                )

        # Format user data
        user_data_str = json.dumps(user_data, indent=2, default=str)

        prompt = f"""Given the following form fields and user data, map the user data to the correct form fields.

FORM FIELDS:
{chr(10).join(field_list)}

CALCULATION RULES (do NOT calculate these - just note them):
{chr(10).join(calc_rules) if calc_rules else "(none)"}

USER DATA:
{user_data_str}

For each field that can be filled from the user data:
1. field_id: The field ID from the list above
2. value: The value to fill (formatted appropriately)
3. confidence: 0.0-1.0 confidence score
4. reasoning: Why this value was chosen
5. source_key: The key in user_data that provided this value

Do NOT fill calculated fields (those with calculation rules).
Do NOT guess at values not in the user data.
Return null for value if no suitable data exists."""

        return prompt


class Mapper:
    """Main mapper class that coordinates data-to-field mapping."""

    def __init__(
        self,
        user_data: dict[str, Any] | str | Path,
        form_schema: FormSchema,
        instruction_map: InstructionMap | None = None,
        llm_config: LLMConfig | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize mapper.

        Args:
            user_data: User data dict or path to JSON file
            form_schema: Form schema with field definitions
            instruction_map: Optional instruction map for guidance
            llm_config: LLM configuration
            verbose: Enable verbose logging
        """
        self.form_schema = form_schema
        self.instruction_map = instruction_map
        self.llm_config = llm_config
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        # Load user data
        self.user_data = self._load_user_data(user_data)

        # Initialize provider if config provided
        self._provider: LLMProvider | None = None

        # Track all mapped values for calculation context
        self._field_values: dict[str, float | int | None] = {}

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

    def _load_user_data(self, data: dict[str, Any] | str | Path) -> dict[str, Any]:
        """Load user data from dict or file."""
        if isinstance(data, dict):
            return data

        path = Path(data)
        if not path.exists():
            raise DataLoadError(f"User data file not found: {path}")

        try:
            content = path.read_text()
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise DataLoadError(f"Invalid JSON in user data file: {e}") from e

    def map(self) -> FieldMappingResult:
        """Map user data to form fields.

        This is the main entry point for the mapper. It:
        1. Uses LLM to map data to non-calculated fields
        2. Executes calculations programmatically
        3. Cross-checks any LLM-mapped calculated fields
        4. Flags low-confidence mappings
        5. Returns complete mapping result

        Returns:
            FieldMappingResult with all mappings and warnings
        """
        logger.info(f"Mapping {len(self.user_data)} data keys to {len(self.form_schema.fields)} fields")

        result = FieldMappingResult()

        # Step 1: Get calculation field IDs
        calc_field_ids = set()
        if self.instruction_map:
            for rule in self.instruction_map.calculation_rules:
                calc_field_ids.add(rule.target)

        # Step 2: Use LLM to map non-calculated fields
        llm_mappings: list[FieldMapping] = []
        if self.provider:
            mapper = DataToFieldMapper(self.provider)
            llm_mappings = mapper.map_data_to_fields(
                self.user_data,
                self.form_schema,
                self.instruction_map,
            )
            logger.info(f"LLM produced {len(llm_mappings)} mappings")

        # Process LLM mappings
        for mapping in llm_mappings:
            # Skip if this is a calculated field (we'll handle those separately)
            if mapping.field_id in calc_field_ids:
                # Store for later cross-check
                self._field_values[mapping.field_id] = self._parse_numeric(mapping.value)
                continue

            # Determine if needs review based on confidence
            needs_review, review_reason = self._check_confidence(
                mapping.confidence,
                mapping.field_id,
            )

            result.mappings.append(FieldMapping(
                field_id=mapping.field_id,
                value=mapping.value,
                confidence=mapping.confidence,
                reasoning=mapping.reasoning,
                source_key=mapping.source_key,
                calculated=False,
            ))

            if needs_review:
                result.warnings.append(MappingWarning(
                    field_id=mapping.field_id,
                    message=review_reason or "Low confidence mapping",
                    severity="warning" if mapping.confidence >= CONFIDENCE_REVIEW else "error",
                ))

            # Track for calculation context
            if mapping.value is not None:
                self._field_values[mapping.field_id] = self._parse_numeric(mapping.value)

        # Step 3: Execute calculations programmatically
        if self.instruction_map:
            for rule in self.instruction_map.calculation_rules:
                # Capture LLM value before calculation overwrites it
                llm_value_before = self._field_values.get(rule.target)

                calc_result = self._execute_calculation(rule)
                if calc_result:
                    result.calculations.append(FieldMapping(
                        field_id=calc_result.field_id,
                        value=str(calc_result.value) if calc_result.value is not None else None,
                        confidence=1.0 if calc_result.verified else 0.85,
                        reasoning=f"Calculated using: {calc_result.formula}",
                        calculated=True,
                        formula=calc_result.formula,
                    ))

                    # Cross-check with LLM mapping if exists
                    if llm_value_before is not None and calc_result.value is not None:
                        if abs(float(llm_value_before) - float(calc_result.value)) > 0.01:
                            result.warnings.append(MappingWarning(
                                field_id=calc_result.field_id,
                                message=f"Calculation mismatch: LLM={llm_value_before}, Calculated={calc_result.value}",
                                severity="warning",
                            ))

        # Step 4: Identify unmapped fields
        mapped_field_ids = {m.field_id for m in result.mappings}
        mapped_field_ids.update({m.field_id for m in result.calculations})

        for field in self.form_schema.fields:
            if field.id not in mapped_field_ids and field.required:
                result.unmapped_fields.append(field.id)
                result.warnings.append(MappingWarning(
                    field_id=field.id,
                    message=f"No data provided for required field '{field.label or field.id}'",
                    severity="info",
                ))

        # Step 5: Identify unmapped user data
        mapped_data_keys = {
            m.source_key for m in result.mappings
            if m.source_key
        }
        for key in self.user_data:
            if key not in mapped_data_keys:
                result.unmapped_data.append(key)

        logger.info(
            f"Mapping complete: {len(result.mappings)} mapped, "
            f"{len(result.calculations)} calculated, "
            f"{len(result.unmapped_fields)} unmapped fields, "
            f"{len(result.warnings)} warnings"
        )

        return result

    def _execute_calculation(self, rule: CalculationRule) -> CalculationResult | None:
        """Execute a calculation rule."""
        executor = CalculationExecutor(self._field_values)
        result = executor.execute(rule.formula)
        result.field_id = rule.target

        if result.error:
            logger.warning(f"Calculation failed for {rule.target}: {result.error}")
            result.verified = False
        else:
            result.verified = True

        # Update field values for subsequent calculations
        if result.value is not None:
            self._field_values[rule.target] = result.value

        return result

    def _check_confidence(
        self,
        confidence: float,
        field_id: str,
    ) -> tuple[bool, str | None]:
        """Check if a confidence score requires review.

        Returns:
            Tuple of (needs_review, reason)
        """
        if confidence < CONFIDENCE_REVIEW:
            return True, f"Very low confidence ({confidence:.2f}) - requires manual input"
        elif confidence < CONFIDENCE_AUTO_FILL:
            return True, f"Low confidence ({confidence:.2f}) - recommend review"
        return False, None

    def _parse_numeric(self, value: str | None) -> float | None:
        """Parse a numeric value from string."""
        if value is None:
            return None
        try:
            # Remove common formatting
            cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    def to_json(self, result: FieldMappingResult | None = None) -> str:
        """Export mapping result as JSON string.

        Args:
            result: Mapping result to export (if None, maps first)

        Returns:
            JSON string
        """
        if result is None:
            result = self.map()

        return result.model_dump_json(indent=2)


def map_data_to_fields(
    user_data: dict[str, Any] | str | Path,
    form_schema: FormSchema,
    instruction_map: InstructionMap | None = None,
    provider: LLMProvider | None = None,
    verbose: bool = False,
) -> FieldMappingResult:
    """Convenience function to map user data to fields.

    Args:
        user_data: User data dict or path to JSON file
        form_schema: Form schema with field definitions
        instruction_map: Optional instruction map
        provider: Optional LLM provider
        verbose: Enable verbose logging

    Returns:
        FieldMappingResult with all mappings
    """
    mapper = Mapper(
        user_data=user_data,
        form_schema=form_schema,
        instruction_map=instruction_map,
        verbose=verbose,
    )
    return mapper.map()
