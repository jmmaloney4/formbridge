"""Tests for the FormBridge mapper module (Phase 3).

Covers:
- User data loading
- LLM-based data-to-field mapping (mocked)
- Programmatic calculation execution
- Cross-checking calculated vs LLM-mapped values
- Confidence scoring
- Flagging low-confidence fields
- MappingWarning and FieldMappingResult construction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from formbridge.mapper import (
    CONFIDENCE_AUTO_FILL,
    CONFIDENCE_REVIEW,
    CalculationError,
    CalculationExecutor,
    CalculationResult,
    DataLoadError,
    DataToFieldMapper,
    Mapper,
    MapperError,
    map_data_to_fields,
)
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
    MappingWarning,
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
                line_ref="A",
                required=True,
            ),
            FormField(
                id="field_002",
                label="EIN",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=350, y=700, w=100, h=15),
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
                required=True,
            ),
            FormField(
                id="field_006",
                label="Start date",
                page=1,
                type=FieldType.DATE,
            ),
            FormField(
                id="field_007",
                label="Agree to terms",
                page=1,
                type=FieldType.CHECKBOX,
            ),
        ],
    )


@pytest.fixture
def sample_instruction_map() -> InstructionMap:
    """Create a sample instruction map with calculation rules."""
    return InstructionMap(
        form_id="test-form",
        field_instructions={
            "field_001": FieldInstruction(
                line_ref="A",
                label="Name",
                instruction="Enter the full legal name.",
                examples=["Smith & Co"],
            ),
            "field_002": FieldInstruction(
                line_ref="B",
                label="EIN",
                instruction="Enter the EIN.",
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
def sample_user_data() -> dict[str, Any]:
    """Create sample user data."""
    return {
        "name": "Acme Partners LLC",
        "ein": "82-1234567",
        "gross_receipts": 10000,
        "deductions": 2000,
        "start_date": "2023-03-15",
        "agree_terms": True,
        "extra_field": "not used",
    }


@pytest.fixture
def llm_mapping_response() -> dict[str, Any]:
    """Create a mock LLM response for data mapping."""
    return {
        "mappings": [
            {
                "field_id": "field_001",
                "value": "Acme Partners LLC",
                "confidence": 0.98,
                "reasoning": "User data 'name' matches Line A for legal name",
                "source_key": "name",
            },
            {
                "field_id": "field_002",
                "value": "82-1234567",
                "confidence": 0.99,
                "reasoning": "EIN matches required format",
                "source_key": "ein",
            },
            {
                "field_id": "field_003",
                "value": "10000",
                "confidence": 0.95,
                "reasoning": "Matches gross receipts field",
                "source_key": "gross_receipts",
            },
            {
                "field_id": "field_004",
                "value": "2000",
                "confidence": 0.95,
                "reasoning": "Matches deductions field",
                "source_key": "deductions",
            },
            {
                "field_id": "field_006",
                "value": "03/15/2023",
                "confidence": 0.92,
                "reasoning": "Date reformatted to US format",
                "source_key": "start_date",
            },
            {
                "field_id": "field_007",
                "value": "Yes",
                "confidence": 0.97,
                "reasoning": "Boolean converted to checkbox value",
                "source_key": "agree_terms",
            },
        ],
        "unmapped_fields": ["field_005"],
        "unmapped_data": ["extra_field"],
    }


def _mock_litellm_response(content: Any, model: str = "gpt-4o-mini") -> MagicMock:
    """Create a mock litellm.completion() response object."""
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


# =============================================================================
# Calculation Executor Tests (Pure Math - No LLM)
# =============================================================================


class TestCalculationExecutor:
    """Tests for programmatic calculation execution."""

    def test_simple_subtraction(self) -> None:
        """Test simple subtraction formula."""
        executor = CalculationExecutor({
            "field_003": 10000,
            "field_004": 2000,
        })
        result = executor.execute("field_003 - field_004")

        assert result.error is None
        assert result.value == 8000
        assert set(result.source_fields) == {"field_003", "field_004"}

    def test_simple_addition(self) -> None:
        """Test simple addition formula."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": 200,
            "field_003": 300,
        })
        result = executor.execute("field_001 + field_002 + field_003")

        assert result.error is None
        assert result.value == 600

    def test_multiplication(self) -> None:
        """Test multiplication formula."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": 0.15,
        })
        result = executor.execute("field_001 * field_002")

        assert result.error is None
        assert result.value == 15.0

    def test_division(self) -> None:
        """Test division formula."""
        executor = CalculationExecutor({
            "field_001": 1000,
            "field_002": 4,
        })
        result = executor.execute("field_001 / field_002")

        assert result.error is None
        assert result.value == 250.0

    def test_complex_formula(self) -> None:
        """Test complex formula with multiple operations."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": 50,
            "field_003": 25,
        })
        result = executor.execute("(field_001 + field_002) - field_003")

        assert result.error is None
        assert result.value == 125

    def test_sum_function(self) -> None:
        """Test sum function."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": 200,
            "field_003": 300,
        })
        result = executor.execute("sum(field_001, field_002, field_003)")

        assert result.error is None
        assert result.value == 600

    def test_max_function(self) -> None:
        """Test max function."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": 500,
            "field_003": 50,
        })
        result = executor.execute("max(field_001, field_002, field_003)")

        assert result.error is None
        assert result.value == 500

    def test_min_function(self) -> None:
        """Test min function."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": 50,
            "field_003": 200,
        })
        result = executor.execute("min(field_001, field_002, field_003)")

        assert result.error is None
        assert result.value == 50

    def test_abs_function(self) -> None:
        """Test abs function."""
        executor = CalculationExecutor({"field_001": -100})
        result = executor.execute("abs(field_001)")

        assert result.error is None
        assert result.value == 100

    def test_round_function(self) -> None:
        """Test round function."""
        executor = CalculationExecutor({"field_001": 123.456})
        result = executor.execute("round(field_001)")

        assert result.error is None
        assert result.value == 123

    def test_missing_field_error(self) -> None:
        """Test error when field value is missing."""
        executor = CalculationExecutor({
            "field_001": 100,
            # field_002 is missing
        })
        result = executor.execute("field_001 - field_002")

        assert result.error is not None
        assert "field_002" in result.error
        assert result.value is None

    def test_null_field_value_error(self) -> None:
        """Test error when field value is None."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": None,
        })
        result = executor.execute("field_001 - field_002")

        assert result.error is not None
        assert "field_002" in result.error

    def test_dangerous_formula_blocked(self) -> None:
        """Test that dangerous formulas are blocked."""
        executor = CalculationExecutor({"field_001": 100})
        result = executor.execute("import os")

        assert result.error is not None
        assert "Dangerous" in result.error

    def test_string_values_converted(self) -> None:
        """Test that string numeric values are converted."""
        executor = CalculationExecutor({
            "field_001": "1,000",  # With comma
            "field_002": "$500",   # With dollar sign
        })
        result = executor.execute("field_001 + field_002")

        assert result.error is None
        assert result.value == 1500

    def test_float_precision(self) -> None:
        """Test float precision (rounded to 2 decimals)."""
        executor = CalculationExecutor({
            "field_001": 100,
            "field_002": 3,
        })
        result = executor.execute("field_001 / field_002")

        assert result.error is None
        assert result.value == 33.33  # Rounded to 2 decimals


# =============================================================================
# Data Loading Tests
# =============================================================================


class TestUserDataLoading:
    """Tests for user data loading."""

    def test_load_dict(self, sample_schema: FormSchema) -> None:
        """Test loading user data from dict."""
        data = {"name": "Test"}
        mapper = Mapper(data, sample_schema)
        assert mapper.user_data == data

    def test_load_from_json_file(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test loading user data from JSON file."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample_user_data))

        mapper = Mapper(str(json_file), sample_schema)
        assert mapper.user_data == sample_user_data

    def test_load_from_path(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test loading user data from Path object."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample_user_data))

        mapper = Mapper(json_file, sample_schema)
        assert mapper.user_data == sample_user_data

    def test_load_missing_file(self, sample_schema: FormSchema) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(DataLoadError, match="not found"):
            Mapper("/nonexistent/file.json", sample_schema)

    def test_load_invalid_json(
        self,
        sample_schema: FormSchema,
        tmp_path: Path,
    ) -> None:
        """Test error when JSON is invalid."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json {")

        with pytest.raises(DataLoadError, match="Invalid JSON"):
            Mapper(str(json_file), sample_schema)


# =============================================================================
# LLM Mapping Tests (Mocked)
# =============================================================================


class TestDataToFieldMapper:
    """Tests for LLM-based data-to-field mapping."""

    def test_map_basic(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
        llm_mapping_response: dict[str, Any],
    ) -> None:
        """Test basic LLM mapping."""
        provider = MagicMock()
        provider.complete.return_value = {"content": llm_mapping_response}

        mapper = DataToFieldMapper(provider)
        result = mapper.map_data_to_fields(sample_user_data, sample_schema)

        assert len(result) > 0

        # Check a specific mapping
        name_mapping = next(m for m in result if m.field_id == "field_001")
        assert name_mapping.value == "Acme Partners LLC"
        assert name_mapping.confidence > 0.9
        assert name_mapping.source_key == "name"

    def test_map_with_instructions(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
        sample_instruction_map: InstructionMap,
        llm_mapping_response: dict[str, Any],
    ) -> None:
        """Test mapping with instruction guidance."""
        provider = MagicMock()
        provider.complete.return_value = {"content": llm_mapping_response}

        mapper = DataToFieldMapper(provider)
        result = mapper.map_data_to_fields(
            sample_user_data,
            sample_schema,
            sample_instruction_map,
        )

        # Verify the prompt included instruction info
        call_args = provider.complete.call_args
        messages = call_args[0][0]
        prompt = messages[-1]["content"]

        # Should mention the field format
        assert "field_001" in prompt or "field_002" in prompt

    def test_handles_string_response(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
        llm_mapping_response: dict[str, Any],
    ) -> None:
        """Test handling when LLM returns string content."""
        provider = MagicMock()
        # Return string instead of dict
        provider.complete.return_value = {
            "content": json.dumps(llm_mapping_response)
        }

        mapper = DataToFieldMapper(provider)
        result = mapper.map_data_to_fields(sample_user_data, sample_schema)

        assert len(result) > 0

    def test_ignores_unknown_field_ids(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test that unknown field IDs are ignored."""
        response_with_unknown = {
            "mappings": [
                {
                    "field_id": "field_999",  # Not in schema
                    "value": "test",
                    "confidence": 0.9,
                },
                {
                    "field_id": "field_001",  # Valid
                    "value": "Test Name",
                    "confidence": 0.95,
                },
            ],
        }

        provider = MagicMock()
        provider.complete.return_value = {"content": response_with_unknown}

        mapper = DataToFieldMapper(provider)
        result = mapper.map_data_to_fields(sample_user_data, sample_schema)

        # Should only have the valid field
        assert len(result) == 1
        assert result[0].field_id == "field_001"

    def test_handles_llm_error(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test handling of LLM errors."""
        provider = MagicMock()
        provider.complete.side_effect = Exception("LLM down")

        mapper = DataToFieldMapper(provider)

        with pytest.raises(MapperError, match="LLM mapping failed"):
            mapper.map_data_to_fields(sample_user_data, sample_schema)


# =============================================================================
# Full Mapper Tests
# =============================================================================


class TestMapper:
    """Tests for the full Mapper class."""

    def test_map_without_llm(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test mapping without LLM provider (returns empty)."""
        mapper = Mapper(
            user_data=sample_user_data,
            form_schema=sample_schema,
            llm_config=None,
        )
        result = mapper.map()

        # Without LLM, should have no LLM mappings
        assert len(result.mappings) == 0
        # But should track unmapped required fields
        assert len(result.unmapped_fields) > 0

    def test_map_with_mock_llm(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
        sample_instruction_map: InstructionMap,
        llm_mapping_response: dict[str, Any],
    ) -> None:
        """Test full mapping pipeline with mocked LLM."""
        from formbridge.llm import LLMConfig

        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_response(
                llm_mapping_response
            )

            mapper = Mapper(
                user_data=sample_user_data,
                form_schema=sample_schema,
                instruction_map=sample_instruction_map,
                llm_config=llm_config,
            )
            result = mapper.map()

        assert isinstance(result, FieldMappingResult)
        assert len(result.mappings) > 0

        # Check that calculation was executed
        # (field_005 is calculated from field_003 - field_004)
        assert len(result.calculations) == 1
        calc = result.calculations[0]
        assert calc.field_id == "field_005"
        assert calc.calculated is True
        # 10000 - 2000 = 8000
        assert float(calc.value) == 8000.0

    def test_confidence_thresholds(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test that confidence thresholds are applied correctly."""
        # Create response with various confidence levels
        response = {
            "mappings": [
                {"field_id": "field_001", "value": "Test", "confidence": 0.98},  # High
                {"field_id": "field_002", "value": "12-3456789", "confidence": 0.88},  # Medium
                {"field_id": "field_003", "value": "100", "confidence": 0.65},  # Low
            ],
        }

        from formbridge.llm import LLMConfig

        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_response(
                response
            )

            mapper = Mapper(
                user_data=sample_user_data,
                form_schema=sample_schema,
                llm_config=llm_config,
            )
            result = mapper.map()

        # Check warnings for low confidence
        assert len(result.warnings) > 0

        # Find the low confidence warning
        low_conf_warnings = [w for w in result.warnings if "0.65" in w.message or "0.88" in w.message]
        assert len(low_conf_warnings) >= 1

    def test_unmapped_fields_tracking(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test that unmapped fields are tracked."""
        # Response that only maps a few fields
        response = {
            "mappings": [
                {"field_id": "field_001", "value": "Test", "confidence": 0.95},
            ],
        }

        from formbridge.llm import LLMConfig

        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_response(
                response
            )

            mapper = Mapper(
                user_data=sample_user_data,
                form_schema=sample_schema,
                llm_config=llm_config,
            )
            result = mapper.map()

        # Should have unmapped required fields
        assert len(result.unmapped_fields) > 0
        # field_005 is required and wasn't mapped
        assert "field_005" in result.unmapped_fields

    def test_unmapped_data_tracking(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test that unmapped user data keys are tracked."""
        response = {
            "mappings": [
                {"field_id": "field_001", "value": "Test", "confidence": 0.95, "source_key": "name"},
            ],
        }

        from formbridge.llm import LLMConfig

        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_response(
                response
            )

            mapper = Mapper(
                user_data=sample_user_data,
                form_schema=sample_schema,
                llm_config=llm_config,
            )
            result = mapper.map()

        # Should have unmapped data keys
        assert len(result.unmapped_data) > 0

    def test_cross_check_calculation(
        self,
        sample_schema: FormSchema,
        sample_instruction_map: InstructionMap,
    ) -> None:
        """Test cross-checking calculated vs LLM-mapped values."""
        # LLM also provides a value for calculated field (should trigger cross-check)
        user_data = {
            "name": "Test",
            "gross_receipts": 10000,
            "deductions": 2000,
            "net_income": 7500,  # Wrong! Should be 8000
        }

        response = {
            "mappings": [
                {"field_id": "field_003", "value": "10000", "confidence": 0.95, "source_key": "gross_receipts"},
                {"field_id": "field_004", "value": "2000", "confidence": 0.95, "source_key": "deductions"},
                {"field_id": "field_005", "value": "7500", "confidence": 0.90, "source_key": "net_income"},
            ],
        }

        from formbridge.llm import LLMConfig

        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_response(
                response
            )

            mapper = Mapper(
                user_data=user_data,
                form_schema=sample_schema,
                instruction_map=sample_instruction_map,
                llm_config=llm_config,
            )
            result = mapper.map()

        # Should have a warning about calculation mismatch
        mismatch_warnings = [
            w for w in result.warnings
            if "mismatch" in w.message.lower()
        ]
        assert len(mismatch_warnings) >= 1

    def test_to_json(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test JSON export."""
        mapper = Mapper(sample_user_data, sample_schema)
        json_str = mapper.to_json()

        data = json.loads(json_str)
        assert "mappings" in data
        assert "unmapped_fields" in data
        assert "warnings" in data


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for map_data_to_fields convenience function."""

    def test_convenience_function(
        self,
        sample_schema: FormSchema,
        sample_user_data: dict[str, Any],
    ) -> None:
        """Test the convenience function works."""
        result = map_data_to_fields(
            user_data=sample_user_data,
            form_schema=sample_schema,
            provider=None,
        )

        assert isinstance(result, FieldMappingResult)


# =============================================================================
# Model Tests
# =============================================================================


class TestFieldMappingResult:
    """Tests for FieldMappingResult model."""

    def test_empty_result(self) -> None:
        """Test creating empty result."""
        result = FieldMappingResult()
        assert result.mappings == []
        assert result.calculations == []
        assert result.warnings == []

    def test_result_with_data(self) -> None:
        """Test result with data."""
        result = FieldMappingResult(
            mappings=[
                FieldMapping(field_id="f1", value="test", confidence=0.95),
            ],
            calculations=[
                FieldMapping(field_id="f2", value="100", confidence=1.0, calculated=True),
            ],
            warnings=[
                MappingWarning(field_id="f1", message="Test warning"),
            ],
        )

        assert len(result.mappings) == 1
        assert len(result.calculations) == 1
        assert len(result.warnings) == 1

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        result = FieldMappingResult(
            mappings=[FieldMapping(field_id="f1", value="test", confidence=0.95)],
        )

        json_str = result.model_dump_json()
        restored = FieldMappingResult.model_validate_json(json_str)

        assert restored.mappings[0].field_id == "f1"


class TestCalculationResult:
    """Tests for CalculationResult model."""

    def test_success_result(self) -> None:
        """Test successful calculation result."""
        result = CalculationResult(
            field_id="f1",
            formula="f2 - f3",
            value=100,
            source_fields=["f2", "f3"],
            verified=True,
        )
        assert result.error is None
        assert result.verified is True

    def test_error_result(self) -> None:
        """Test calculation result with error."""
        result = CalculationResult(
            field_id="f1",
            formula="f2 - f3",
            value=None,
            error="Missing field: f3",
        )
        assert result.value is None
        assert "f3" in result.error
