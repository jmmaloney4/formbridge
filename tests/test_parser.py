"""Tests for the FormBridge parser module (Phase 2).

Covers:
- Instruction text extraction from PDFs
- LLM provider abstraction (mock HTTP calls)
- Calculation rule extraction
- InstructionMap construction
- Caching
- CLI parse command
- Integration test with real IRS instruction PDF
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from formbridge.llm import (
    LLMAPIError,
    LLMConfig,
    LLMConfigError,
    LLMError,
    LiteLLMProvider,
    create_provider,
    load_config,
)
from formbridge.models import (
    CalculationRule,
    FieldInstruction,
    FieldPosition,
    FieldType,
    FormField,
    FormSchema,
    InstructionMap,
)
from formbridge.parser import (
    InstructionCache,
    InstructionExtractor,
    InstructionLLMMapper,
    Parser,
    ParserError,
    TextSection,
    parse_instructions,
)

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_schema() -> FormSchema:
    """Create a sample form schema with fields for testing."""
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
                line_ref="1a",
            ),
            FormField(
                id="field_004",
                label="Returns and allowances",
                page=1,
                type=FieldType.NUMBER,
                line_ref="1b",
            ),
            FormField(
                id="field_005",
                label="Net receipts",
                page=1,
                type=FieldType.NUMBER,
                line_ref="1c",
            ),
            FormField(
                id="field_006",
                label="Cost of goods sold",
                page=1,
                type=FieldType.NUMBER,
                line_ref="2",
            ),
            FormField(
                id="field_007",
                label="Gross profit",
                page=1,
                type=FieldType.NUMBER,
                line_ref="3",
            ),
            FormField(
                id="field_008",
                label="Total income",
                page=1,
                type=FieldType.NUMBER,
                line_ref="8",
            ),
            FormField(
                id="field_009",
                label="Checkbox example",
                page=2,
                type=FieldType.CHECKBOX,
            ),
        ],
    )


@pytest.fixture
def sample_sections() -> list[TextSection]:
    """Create sample instruction sections for testing."""
    return [
        TextSection(
            heading="Line A",
            content="Enter the legal name of the partnership as shown on the partnership agreement.",
            page_number=4,
            level=2,
        ),
        TextSection(
            heading="Line B",
            content="Enter the nine-digit EIN assigned to the partnership. Format: XX-XXXXXXX.",
            page_number=4,
            level=2,
        ),
        TextSection(
            heading="Line 1a",
            content="Enter gross receipts or sales from all business operations.",
            page_number=5,
            level=2,
        ),
        TextSection(
            heading="Line 3",
            content="Subtract line 2 from line 1c. This is the gross profit.",
            page_number=5,
            level=2,
        ),
        TextSection(
            heading="Line 8",
            content="Add lines 3 through 7. Enter the total income (loss).",
            page_number=6,
            level=2,
        ),
    ]


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM config."""
    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key-123",
        base_url="https://api.openai.com/v1",
    )


@pytest.fixture
def mock_mapping_response() -> dict[str, Any]:
    """Create a mock LLM response for instruction mapping."""
    return {
        "mappings": [
            {
                "field_id": "field_001",
                "line_ref": "A",
                "label": "Name of partnership",
                "instruction": "Enter the legal name of the partnership.",
                "examples": ["Smith & Jones LLP"],
                "constraints": ["Must match EIN registration"],
                "format": None,
                "source_page": 4,
            },
            {
                "field_id": "field_002",
                "line_ref": "B",
                "label": "EIN",
                "instruction": "Enter the nine-digit EIN.",
                "examples": [],
                "constraints": [],
                "format": "XX-XXXXXXX",
                "source_page": 4,
            },
        ],
    }


@pytest.fixture
def mock_calc_response() -> dict[str, Any]:
    """Create a mock LLM response for calculation rules."""
    return {
        "rules": [
            {
                "target_field_id": "field_007",
                "target_line_ref": "3",
                "formula": "field_005 - field_006",
                "description": "Subtract line 2 from line 1c",
            },
        ],
    }


def _mock_litellm_response(content: Any, model: str = "gpt-4o-mini") -> MagicMock:
    """Create a mock litellm completion response."""
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
# Model Tests
# =============================================================================


class TestInstructionModels:
    """Tests for instruction-related Pydantic models."""

    def test_field_instruction_creation(self) -> None:
        """Test creating a FieldInstruction."""
        inst = FieldInstruction(
            line_ref="A",
            label="Name of partnership",
            instruction="Enter the legal name.",
            examples=["Smith & Jones LLP"],
            constraints=["Must match EIN registration"],
            format=None,
            source_page=4,
        )
        assert inst.line_ref == "A"
        assert inst.label == "Name of partnership"
        assert len(inst.examples) == 1

    def test_field_instruction_minimal(self) -> None:
        """Test creating a FieldInstruction with minimal data."""
        inst = FieldInstruction()
        assert inst.line_ref is None
        assert inst.instruction is None
        assert inst.examples is None

    def test_calculation_rule(self) -> None:
        """Test creating a CalculationRule."""
        rule = CalculationRule(
            target="field_007",
            line_ref="3",
            formula="field_005 - field_006",
            description="Subtract line 2 from line 1c",
        )
        assert rule.target == "field_007"
        assert rule.formula == "field_005 - field_006"

    def test_instruction_map(self) -> None:
        """Test creating an InstructionMap."""
        imap = InstructionMap(
            form_id="test-form",
            field_instructions={
                "field_001": FieldInstruction(
                    instruction="Test instruction",
                ),
            },
            calculation_rules=[
                CalculationRule(
                    target="field_007",
                    formula="field_005 - field_006",
                ),
            ],
        )
        assert imap.form_id == "test-form"
        assert len(imap.field_instructions) == 1
        assert len(imap.calculation_rules) == 1

    def test_instruction_map_json_roundtrip(self) -> None:
        """Test InstructionMap JSON serialization roundtrip."""
        imap = InstructionMap(
            form_id="test-form",
            field_instructions={
                "field_001": FieldInstruction(
                    line_ref="A",
                    label="Name",
                    instruction="Enter the name.",
                ),
            },
            calculation_rules=[
                CalculationRule(
                    target="field_007",
                    formula="field_005 - field_006",
                    description="Subtract line 2 from line 1c",
                ),
            ],
        )

        json_str = imap.model_dump_json()
        restored = InstructionMap.model_validate_json(json_str)

        assert restored.form_id == imap.form_id
        assert len(restored.field_instructions) == 1
        assert restored.field_instructions["field_001"].line_ref == "A"
        assert len(restored.calculation_rules) == 1


# =============================================================================
# LLM Config Tests
# =============================================================================


class TestLLMConfig:
    """Tests for LLM configuration loading."""

    def test_default_config(self) -> None:
        """Test default LLM configuration."""
        cfg = LLMConfig(provider="openai")
        assert cfg.provider == "openai"
        assert cfg.effective_model == "gpt-4o-mini"

    def test_anthropic_config(self) -> None:
        """Test Anthropic configuration defaults."""
        cfg = LLMConfig(provider="anthropic")
        assert "claude" in cfg.effective_model

    def test_custom_model(self) -> None:
        """Test custom model override."""
        cfg = LLMConfig(provider="openai", model="gpt-4o")
        assert cfg.effective_model == "gpt-4o"

    def test_load_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading config from environment variables."""
        monkeypatch.setenv("FORMBRIDGE_PROVIDER", "anthropic")
        monkeypatch.setenv("FORMBRIDGE_MODEL", "claude-haiku-4")
        monkeypatch.setenv("FORMBRIDGE_API_KEY", "sk-ant-test")
        monkeypatch.setenv("FORMBRIDGE_API_BASE", "https://custom.api.com")

        cfg = load_config()
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-haiku-4"
        assert cfg.api_key == "sk-ant-test"
        assert cfg.base_url == "https://custom.api.com"

    def test_load_config_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that explicit arguments override env vars."""
        monkeypatch.setenv("FORMBRIDGE_PROVIDER", "anthropic")

        cfg = load_config(provider="openai")
        assert cfg.provider == "openai"

    def test_load_config_from_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading config from a TOML file."""
        toml_file = tmp_path / "formbridge.toml"
        toml_file.write_text(
            '[llm]\nprovider = "anthropic"\nmodel = "claude-haiku-4"\n'
        )

        # Ensure we don't pick up env vars
        monkeypatch.delenv("FORMBRIDGE_PROVIDER", raising=False)
        monkeypatch.delenv("FORMBRIDGE_MODEL", raising=False)

        cfg = load_config(config_path=str(toml_file))
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-haiku-4"

    def test_load_config_toml_with_api_key_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test TOML api_key_env feature for indirect key loading."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.delenv("FORMBRIDGE_API_KEY", raising=False)
        toml_file = tmp_path / "formbridge.toml"
        toml_file.write_text(
            '[llm]\nprovider = "openai"\napi_key_env = "OPENAI_API_KEY"\n'
        )

        cfg = load_config(config_path=str(toml_file))
        assert cfg.api_key == "sk-from-env"


# =============================================================================
# LLM Provider Tests (mocked HTTP)
# =============================================================================


class TestLiteLLMProvider:
    """Tests for LiteLLMProvider with mocked litellm.completion calls."""

    def test_complete_basic(self) -> None:
        """Test basic completion request."""
        with patch(
            "litellm.completion",
            return_value=_mock_litellm_response("Hello, world!"),
        ):
            provider = LiteLLMProvider(
                config=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test-key-123")
            )
            result = provider.complete([{"role": "user", "content": "Hi"}])

        assert result["content"] == "Hello, world!"

    def test_complete_with_schema(self) -> None:
        """Test completion with JSON schema enforcement."""
        json_content = {"mappings": [], "rules": []}

        with patch(
            "litellm.completion",
            return_value=_mock_litellm_response(json_content),
        ) as mock_completion:
            provider = LiteLLMProvider(
                config=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test-key-123")
            )
            result = provider.complete(
                [{"role": "user", "content": "parse this"}],
                schema={"type": "object", "properties": {}},
            )

            # Verify response_format was included in the kwargs
            call_kwargs = mock_completion.call_args
            assert "response_format" in call_kwargs.kwargs or "response_format" in call_kwargs[1]
            rf = call_kwargs.kwargs.get("response_format") or call_kwargs[1].get("response_format")
            assert rf["type"] == "json_schema"

        # Content should be parsed to dict when schema is present
        assert isinstance(result["content"], dict)

    def test_complete_handles_api_error(self) -> None:
        """Test handling of API errors."""
        import litellm

        with patch(
            "litellm.completion",
            side_effect=litellm.RateLimitError(
                message="Rate limit exceeded",
                llm_provider="openai",
                model="gpt-4o-mini",
            ),
        ):
            provider = LiteLLMProvider(
                config=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test-key-123")
            )

            with pytest.raises(LLMAPIError):
                provider.complete([{"role": "user", "content": "Hi"}])

    def test_complete_handles_timeout(self) -> None:
        """Test handling of timeout errors."""
        import litellm

        with patch(
            "litellm.completion",
            side_effect=litellm.Timeout(message="Request timed out", model="gpt-4o-mini", llm_provider="openai"),
        ):
            provider = LiteLLMProvider(
                config=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test-key-123")
            )

            with pytest.raises(LLMAPIError, match="timed out"):
                provider.complete([{"role": "user", "content": "Hi"}])

    def test_complete_handles_auth_error(self) -> None:
        """Test handling of authentication errors."""
        import litellm

        with patch(
            "litellm.completion",
            side_effect=litellm.AuthenticationError(
                message="Invalid API key",
                llm_provider="openai",
                model="gpt-4o-mini",
            ),
        ):
            provider = LiteLLMProvider(
                config=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="bad-key")
            )

            with pytest.raises(LLMConfigError):
                provider.complete([{"role": "user", "content": "Hi"}])


class TestProviderFactory:
    """Tests for the provider factory function."""

    def test_create_openai_provider(self) -> None:
        """Test creating an OpenAI provider."""
        provider = create_provider(provider="openai", api_key="test-key")
        assert isinstance(provider, LiteLLMProvider)

    def test_create_anthropic_provider(self) -> None:
        """Test creating an Anthropic provider."""
        provider = create_provider(provider="anthropic", api_key="test-key")
        assert isinstance(provider, LiteLLMProvider)

    def test_create_local_provider(self) -> None:
        """Test that 'local' creates a LiteLLMProvider with Ollama defaults."""
        provider = create_provider(
            provider="local",
            base_url="http://localhost:11434/v1",
        )
        assert isinstance(provider, LiteLLMProvider)


# =============================================================================
# Text Extraction Tests
# =============================================================================


class TestInstructionExtractor:
    """Tests for InstructionExtractor."""

    def test_extractor_nonexistent_file(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(ParserError, match="not found"):
            InstructionExtractor("/nonexistent/file.pdf")

    def test_extract_from_minimal_pdf(self) -> None:
        """Test extraction from a minimal PDF."""
        minimal_pdf = FIXTURES_DIR / "minimal.pdf"
        if not minimal_pdf.exists():
            pytest.skip("Minimal PDF fixture not available")

        extractor = InstructionExtractor(minimal_pdf)
        sections = extractor.extract_sections()
        assert isinstance(sections, list)

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "f1065.pdf").exists(),
        reason="Form 1065 fixture not available",
    )
    def test_extract_from_irs_form(self) -> None:
        """Test extraction from a real IRS form."""
        extractor = InstructionExtractor(FIXTURES_DIR / "f1065.pdf")
        sections = extractor.extract_sections()
        assert len(sections) > 0

        # Check that sections have text
        for section in sections:
            assert isinstance(section.content, str)
            assert section.page_number >= 1

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "f1065.pdf").exists(),
        reason="Form 1065 fixture not available",
    )
    def test_extract_text_by_line_reference(self) -> None:
        """Test extracting text organized by line reference."""
        extractor = InstructionExtractor(FIXTURES_DIR / "f1065.pdf")
        line_sections = extractor.extract_text_by_line_reference()
        assert isinstance(line_sections, dict)

    def test_detect_heading_level(self) -> None:
        """Test heading level detection."""
        extractor = InstructionExtractor.__new__(InstructionExtractor)

        # Line reference patterns should be detected
        assert extractor._detect_heading_level("Line 1. Enter gross receipts") == 2
        assert extractor._detect_heading_level("Line 16a") == 2

        # Numbered headings
        assert extractor._detect_heading_level("1. Introduction") >= 1

        # Regular text should not be heading
        assert extractor._detect_heading_level("Enter the total amount") == 0

    def test_extract_line_ref(self) -> None:
        """Test line reference extraction from text."""
        extractor = InstructionExtractor.__new__(InstructionExtractor)

        assert extractor._extract_line_ref_from_text("Line 16a") == "16a"
        assert extractor._extract_line_ref_from_text("Line 1") == "1"
        assert extractor._extract_line_ref_from_text("A. Name of partnership") == "A"
        assert extractor._extract_line_ref_from_text("(B) EIN") == "B"
        assert extractor._extract_line_ref_from_text("Some random text") is None


class TestTextSection:
    """Tests for TextSection dataclass."""

    def test_section_creation(self) -> None:
        """Test creating a TextSection."""
        section = TextSection(
            heading="Line A",
            content="Enter the name.",
            page_number=4,
            level=2,
        )
        assert section.heading == "Line A"
        assert section.content == "Enter the name."
        assert section.page_number == 4
        assert section.level == 2


# =============================================================================
# InstructionLLMMapper Tests
# =============================================================================


class TestInstructionLLMMapper:
    """Tests for LLM-powered instruction mapping."""

    def _mock_provider(self, response_content: dict) -> MagicMock:
        """Create a mock LLM provider that returns given content."""
        provider = MagicMock()
        provider.complete.return_value = {"content": response_content}
        return provider

    def test_map_instructions_to_fields(
        self,
        sample_sections: list[TextSection],
        sample_schema: FormSchema,
        mock_mapping_response: dict[str, Any],
    ) -> None:
        """Test mapping instructions to fields via LLM."""
        provider = self._mock_provider(mock_mapping_response)
        mapper = InstructionLLMMapper(provider)

        result = mapper.map_instructions_to_fields(sample_sections, sample_schema)

        assert isinstance(result, dict)
        assert "field_001" in result
        assert result["field_001"].line_ref == "A"
        assert result["field_001"].label == "Name of partnership"
        assert result["field_002"].format == "XX-XXXXXXX"

    def test_ignores_unknown_field_ids(
        self,
        sample_sections: list[TextSection],
        sample_schema: FormSchema,
    ) -> None:
        """Test that mapper ignores matches with unknown field IDs."""
        response = {
            "mappings": [
                {
                    "field_id": "field_999",  # Not in schema
                    "instruction": "This should be ignored",
                },
                {
                    "field_id": "field_001",  # Valid
                    "instruction": "This should be kept",
                },
            ],
        }
        provider = self._mock_provider(response)
        mapper = InstructionLLMMapper(provider)

        result = mapper.map_instructions_to_fields(sample_sections, sample_schema)

        assert "field_001" in result
        assert "field_999" not in result

    def test_handles_string_response(
        self,
        sample_sections: list[TextSection],
        sample_schema: FormSchema,
        mock_mapping_response: dict[str, Any],
    ) -> None:
        """Test handling when LLM returns a string instead of dict."""
        provider = MagicMock()
        # Return string content (provider didn't parse JSON)
        provider.complete.return_value = {"content": json.dumps(mock_mapping_response)}
        mapper = InstructionLLMMapper(provider)

        result = mapper.map_instructions_to_fields(sample_sections, sample_schema)
        assert "field_001" in result

    def test_extract_calculation_rules(
        self,
        sample_sections: list[TextSection],
        sample_schema: FormSchema,
        mock_calc_response: dict[str, Any],
    ) -> None:
        """Test extracting calculation rules via LLM."""
        provider = self._mock_provider(mock_calc_response)
        mapper = InstructionLLMMapper(provider)

        rules = mapper.extract_calculation_rules(sample_sections, sample_schema)

        assert len(rules) >= 1
        rule = rules[0]
        assert rule.target == "field_007"
        assert rule.line_ref == "3"
        assert "field_005" in rule.formula
        assert "field_006" in rule.formula

    def test_calc_rules_with_line_ref_fallback(
        self,
        sample_sections: list[TextSection],
        sample_schema: FormSchema,
    ) -> None:
        """Test calculation rules use line_ref-to-field_id mapping."""
        response = {
            "rules": [
                {
                    "target_line_ref": "3",
                    # No target_field_id - should be resolved from line_ref
                    "formula": "field_005 - field_006",
                    "description": "Subtract line 2 from line 1c",
                },
            ],
        }
        provider = self._mock_provider(response)
        mapper = InstructionLLMMapper(provider)

        rules = mapper.extract_calculation_rules(sample_sections, sample_schema)

        assert len(rules) == 1
        # Should have resolved line_ref "3" -> field_007
        assert rules[0].target == "field_007"

    def test_handles_llm_failure(
        self,
        sample_sections: list[TextSection],
        sample_schema: FormSchema,
    ) -> None:
        """Test that mapper raises on LLM failure."""
        provider = MagicMock()
        provider.complete.side_effect = Exception("LLM is down")
        mapper = InstructionLLMMapper(provider)

        from formbridge.parser import LLMMappingError

        with pytest.raises(LLMMappingError):
            mapper.map_instructions_to_fields(sample_sections, sample_schema)


# =============================================================================
# Cache Tests
# =============================================================================


class TestInstructionCache:
    """Tests for instruction caching."""

    def test_cache_miss(self, tmp_path: Path) -> None:
        """Test cache returns None on miss."""
        cache = InstructionCache(cache_dir=tmp_path / "cache")
        result = cache.get("/some/file.pdf", "form-id")
        assert result is None

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        """Test storing and retrieving from cache."""
        cache = InstructionCache(cache_dir=tmp_path / "cache")

        imap = InstructionMap(
            form_id="test-form",
            field_instructions={
                "field_001": FieldInstruction(instruction="Test"),
            },
            calculation_rules=[],
        )

        # Create a fake file so the cache key is stable
        fake_pdf = tmp_path / "instructions.pdf"
        fake_pdf.write_bytes(b"fake pdf content")

        cache.set(str(fake_pdf), "test-form", imap)
        result = cache.get(str(fake_pdf), "test-form")

        assert result is not None
        assert result.form_id == "test-form"
        assert "field_001" in result.field_instructions

    def test_cache_clear(self, tmp_path: Path) -> None:
        """Test clearing cache."""
        cache = InstructionCache(cache_dir=tmp_path / "cache")

        fake_pdf = tmp_path / "instructions.pdf"
        fake_pdf.write_bytes(b"fake pdf content")

        imap = InstructionMap(form_id="test", field_instructions={}, calculation_rules=[])
        cache.set(str(fake_pdf), "test", imap)

        # Verify it's cached
        assert cache.get(str(fake_pdf), "test") is not None

        # Clear
        cache.clear()

        # Should be gone
        assert cache.get(str(fake_pdf), "test") is None


# =============================================================================
# Parser Integration Tests
# =============================================================================


class TestParser:
    """Tests for the main Parser class."""

    def test_parser_nonexistent_file(self) -> None:
        """Test parser raises error for nonexistent file."""
        with pytest.raises(ParserError, match="not found"):
            Parser("/nonexistent.pdf")

    def test_parser_requires_schema(self, tmp_path: Path) -> None:
        """Test parser requires schema for parsing."""
        pdf = tmp_path / "test.pdf"
        _create_test_instruction_pdf(pdf)

        parser = Parser(instructions_path=pdf, schema=None, use_cache=False)
        with pytest.raises(ParserError, match="schema"):
            parser.parse()

    def test_parser_no_provider_returns_empty(
        self,
        sample_schema: FormSchema,
        tmp_path: Path,
    ) -> None:
        """Test parser without LLM provider returns empty map."""
        pdf = tmp_path / "test.pdf"
        _create_test_instruction_pdf(pdf)

        parser = Parser(
            instructions_path=pdf,
            schema=sample_schema,
            llm_config=None,
            use_cache=False,
        )
        result = parser.parse()

        assert isinstance(result, InstructionMap)
        assert result.form_id == "test-form-1065"
        # Without a provider, no field instructions should be mapped
        assert len(result.field_instructions) == 0

    def test_parser_with_mock_llm(
        self,
        sample_schema: FormSchema,
        mock_mapping_response: dict[str, Any],
        mock_calc_response: dict[str, Any],
        llm_config: LLMConfig,
        tmp_path: Path,
    ) -> None:
        """Test full parse pipeline with mocked LLM."""
        pdf = tmp_path / "instructions.pdf"
        _create_test_instruction_pdf(pdf)

        # LLM will be called twice: once for mapping, once for calc rules
        call_count = [0]
        responses = [
            _mock_litellm_response(mock_mapping_response),
            _mock_litellm_response(mock_calc_response),
        ]

        def mock_completion(**kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        with patch("litellm.completion", mock_completion):
            parser = Parser(
                instructions_path=pdf,
                schema=sample_schema,
                llm_config=llm_config,
                use_cache=False,
            )
            result = parser.parse()

        assert isinstance(result, InstructionMap)
        assert result.form_id == "test-form-1065"
        assert "field_001" in result.field_instructions
        assert result.field_instructions["field_001"].line_ref == "A"

    def test_parser_caching(
        self,
        sample_schema: FormSchema,
        mock_mapping_response: dict[str, Any],
        mock_calc_response: dict[str, Any],
        llm_config: LLMConfig,
        tmp_path: Path,
    ) -> None:
        """Test that parser caches results."""
        pdf = tmp_path / "instructions.pdf"
        _create_test_instruction_pdf(pdf)

        call_count = [0]
        responses = [
            _mock_litellm_response(mock_mapping_response),
            _mock_litellm_response(mock_calc_response),
        ]

        def mock_completion(**kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        # Use a custom cache dir to avoid polluting real cache
        cache_dir = tmp_path / "cache"

        with patch("litellm.completion", mock_completion):
            with patch.object(InstructionCache, "__init__", lambda self, cache_dir=None: (
                setattr(self, 'cache_dir', cache_dir or Path(str(tmp_path / "cache"))) or
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            )):
                # First parse - should call LLM
                parser = Parser(
                    instructions_path=pdf,
                    schema=sample_schema,
                    llm_config=llm_config,
                    use_cache=True,
                )
                result1 = parser.parse()
                first_call_count = call_count[0]

                # Second parse - should use cache
                parser2 = Parser(
                    instructions_path=pdf,
                    schema=sample_schema,
                    llm_config=llm_config,
                    use_cache=True,
                )
                result2 = parser2.parse()

        assert first_call_count > 0
        assert call_count[0] == first_call_count  # No additional LLM calls
        assert result1.form_id == result2.form_id

    def test_parser_to_json(
        self,
        sample_schema: FormSchema,
        mock_mapping_response: dict[str, Any],
        mock_calc_response: dict[str, Any],
        llm_config: LLMConfig,
        tmp_path: Path,
    ) -> None:
        """Test JSON export from parser."""
        pdf = tmp_path / "instructions.pdf"
        _create_test_instruction_pdf(pdf)

        responses = [
            _mock_litellm_response(mock_mapping_response),
            _mock_litellm_response(mock_calc_response),
        ]
        call_idx = [0]

        def mock_completion(**kwargs):
            idx = min(call_idx[0], len(responses) - 1)
            call_idx[0] += 1
            return responses[idx]

        with patch("litellm.completion", mock_completion):
            parser = Parser(
                instructions_path=pdf,
                schema=sample_schema,
                llm_config=llm_config,
                use_cache=False,
            )
            json_str = parser.to_json()

        data = json.loads(json_str)
        assert "form_id" in data
        assert "field_instructions" in data
        assert "calculation_rules" in data


# =============================================================================
# CLI Tests
# =============================================================================


class TestParseCLI:
    """Tests for the parse CLI command."""

    def test_parse_help(self) -> None:
        """Test parse command help output."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["parse", "--help"])

        assert result.exit_code == 0
        assert "Parse instruction document" in result.output
        assert "--fields" in result.output
        assert "--output" in result.output

    def test_parse_missing_instructions_file(self) -> None:
        """Test parse with missing instructions file."""
        from click.testing import CliRunner
        from formbridge.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["parse", "/nonexistent.pdf"])

        assert result.exit_code != 0

    def test_parse_missing_fields(self, tmp_path: Path) -> None:
        """Test parse without --fields flag."""
        from click.testing import CliRunner
        from formbridge.cli import main

        # Create a dummy PDF
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")

        runner = CliRunner()
        result = runner.invoke(main, ["parse", str(pdf_file)])

        # Should require --fields
        assert result.exit_code != 0 or "--fields" in result.output or "error" in result.output.lower()

    def test_parse_end_to_end_mock(
        self,
        sample_schema: FormSchema,
        mock_mapping_response: dict[str, Any],
        mock_calc_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test parse command end-to-end with mocked LLM."""
        from click.testing import CliRunner
        from formbridge.cli import main

        # Create a test instruction PDF
        pdf_path = tmp_path / "instructions.pdf"
        _create_test_instruction_pdf(pdf_path)

        # Write schema to file
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(sample_schema.model_dump_json(indent=2))

        # Output file
        output_path = tmp_path / "instructions.json"

        responses = [
            _mock_litellm_response(mock_mapping_response),
            _mock_litellm_response(mock_calc_response),
        ]
        call_idx = [0]

        def mock_completion(**kwargs):
            idx = min(call_idx[0], len(responses) - 1)
            call_idx[0] += 1
            return responses[idx]

        with patch("litellm.completion", mock_completion):
            runner = CliRunner()
            result = runner.invoke(main, [
                "--provider", "openai",
                "parse", str(pdf_path),
                "--fields", str(schema_path),
                "--output", str(output_path),
                "--no-cache",
            ], env={"FORMBRIDGE_API_KEY": "test-key"})

        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert output_path.exists()

        # Verify output is valid InstructionMap JSON
        data = json.loads(output_path.read_text())
        assert "form_id" in data
        assert "field_instructions" in data


# =============================================================================
# Integration Tests (real IRS PDFs)
# =============================================================================


@pytest.mark.skipif(
    not (FIXTURES_DIR / "f1065.pdf").exists(),
    reason="Form 1065 fixture not available. Run: python tests/download_fixtures.py",
)
class TestParserWithIRSForm:
    """Integration tests using real IRS Form 1065."""

    def test_extract_sections_from_irs_form(self) -> None:
        """Test section extraction from the actual IRS form."""
        extractor = InstructionExtractor(FIXTURES_DIR / "f1065.pdf")
        sections = extractor.extract_sections()
        assert len(sections) > 0

        # Should have content from multiple pages
        pages_covered = {s.page_number for s in sections}
        assert len(pages_covered) > 1

    def test_line_reference_extraction_from_irs_form(self) -> None:
        """Test line reference extraction from the IRS form."""
        extractor = InstructionExtractor(FIXTURES_DIR / "f1065.pdf")
        line_sections = extractor.extract_text_by_line_reference()
        assert isinstance(line_sections, dict)
        # May or may not find line refs in the form itself (vs instructions)

    def test_field_schema_loads(self) -> None:
        """Test that the scanned f1065 schema can be loaded."""
        schema_path = FIXTURES_DIR / "f1065_schema.json"
        if not schema_path.exists():
            pytest.skip("Schema not generated. Run: formbridge scan tests/fixtures/f1065.pdf --output tests/fixtures/f1065_schema.json")

        schema = FormSchema.model_validate_json(schema_path.read_text())
        assert len(schema.fields) > 0
        assert schema.pages > 0


# =============================================================================
# Helpers
# =============================================================================


def _create_test_instruction_pdf(path: Path) -> None:
    """Create a minimal but valid PDF with instruction-like text.

    Uses reportlab for a proper PDF that pdfplumber can parse.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(path), pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 750, "Instructions for Form 1065")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, 720, "Line A.")
        c.setFont("Helvetica", 10)
        c.drawString(72, 705, "Enter the legal name of the partnership.")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, 680, "Line B.")
        c.setFont("Helvetica", 10)
        c.drawString(72, 665, "Enter the nine-digit EIN. Format: XX-XXXXXXX.")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, 640, "Line 3.")
        c.setFont("Helvetica", 10)
        c.drawString(72, 625, "Subtract line 2 from line 1c.")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, 600, "Line 8.")
        c.setFont("Helvetica", 10)
        c.drawString(72, 585, "Add lines 3 through 7.")
        c.save()
    except ImportError:
        # Fallback: minimal PDF
        content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 120>>stream
BT /F1 12 Tf 72 720 Td (Line A. Enter the partnership name.) Tj 0 -20 Td (Line B. Enter the EIN.) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000438 00000 n 
trailer<</Root 1 0 R/Size 6>>
startxref
521
%%EOF"""
        path.write_bytes(content)
