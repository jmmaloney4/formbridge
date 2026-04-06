"""Tests for the vision label extraction module (ADR 001)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from formbridge.models import FieldPosition, FieldType, FormField
from formbridge.vision import (
    LLMVisionLabelProvider,
    build_field_list_text,
    build_vision_prompt,
    pdf_to_pixel,
    render_annotated_page,
)


class TestCoordinateConversion:
    """Test PDF-to-pixel coordinate conversion."""

    def test_top_left(self):
        """PDF origin (0, height) maps to pixel (0, 0)."""
        px, py = pdf_to_pixel(0, 792, 612, 792, 1700, 2200)
        assert px == 0
        assert py == 0

    def test_bottom_left(self):
        """PDF (0, 0) maps to pixel (0, img_height)."""
        px, py = pdf_to_pixel(0, 0, 612, 792, 1700, 2200)
        assert px == 0
        assert abs(py - 2200) < 1

    def test_bottom_right(self):
        """PDF (width, 0) maps to pixel (img_width, img_height)."""
        px, py = pdf_to_pixel(612, 0, 612, 792, 1700, 2200)
        assert abs(px - 1700) < 1
        assert abs(py - 2200) < 1

    def test_midpoint(self):
        """PDF center maps to pixel center."""
        px, py = pdf_to_pixel(306, 396, 612, 792, 1700, 2200)
        assert abs(px - 850) < 1
        assert abs(py - 1100) < 1

    def test_scale_independent(self):
        """Conversion works with any page/image size ratio."""
        px, py = pdf_to_pixel(100, 700, 612, 792, 500, 600)
        expected_px = 100 * (500 / 612)
        expected_py = (792 - 700) * (600 / 792)
        assert abs(px - expected_px) < 0.1
        assert abs(py - expected_py) < 0.1


class TestPromptBuilding:
    """Test vision prompt construction."""

    def test_field_list_text(self):
        fields = [
            FormField(
                id="field_001",
                label="First name",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=33.8, y=183.0, w=200, h=12),
            ),
            FormField(
                id="field_002",
                label=None,
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=33.8, y=213.0, w=200, h=12),
            ),
        ]
        text = build_field_list_text(fields)
        assert "[1] field_001 (text)" in text
        assert "(current: First name)" in text
        assert "[2] field_002 (text)" in text

    def test_vision_prompt_contains_json_instruction(self):
        fields = [
            FormField(id="field_001", page=1, type=FieldType.TEXT),
        ]
        prompt = build_vision_prompt(fields)
        assert "JSON object" in prompt
        assert "field_001" in prompt


class TestLLMVisionLabelProvider:
    """Test the LLM-based vision label provider."""

    def _make_fields(self, count: int = 3) -> list[FormField]:
        return [
            FormField(
                id=f"field_{i:03d}",
                page=1,
                type=FieldType.TEXT,
                position=FieldPosition(x=10 * i, y=100, w=200, h=12),
            )
            for i in range(1, count + 1)
        ]

    def test_refine_labels_success(self):
        """Vision provider returns labels from LLM response."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "content": {
                "labels": {
                    "1": "First name and middle initial",
                    "2": "Last name",
                    "3": "Social security number",
                },
                "confidence": {"1": 0.95, "2": 0.92, "3": 0.88},
            },
        }

        vision = LLMVisionLabelProvider(mock_provider)
        fields = self._make_fields(3)
        result = vision.refine_labels(b"fake_png_bytes", fields, 612, 792)

        assert len(result) == 3
        assert result["field_001"] == ("First name and middle initial", 0.95)
        assert result["field_002"] == ("Last name", 0.92)
        assert result["field_003"] == ("Social security number", 0.88)

    def test_refine_labels_partial_response(self):
        """Handles partial LLM responses (not all fields labeled)."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "content": {
                "labels": {"1": "First name"},
                "confidence": {"1": 0.9},
            },
        }

        vision = LLMVisionLabelProvider(mock_provider)
        fields = self._make_fields(3)
        result = vision.refine_labels(b"fake_png_bytes", fields, 612, 792)

        assert len(result) == 1
        assert result["field_001"] == ("First name", 0.9)

    def test_refine_labels_llm_error(self):
        """Returns empty dict when LLM call fails."""
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = Exception("API error")

        vision = LLMVisionLabelProvider(mock_provider)
        fields = self._make_fields(3)
        result = vision.refine_labels(b"fake_png_bytes", fields, 612, 792)

        assert result == {}

    def test_refine_labels_string_response(self):
        """Handles LLM returning string instead of parsed JSON."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "content": json.dumps({
                "labels": {"1": "Wages"},
                "confidence": {"1": 0.85},
            }),
        }

        vision = LLMVisionLabelProvider(mock_provider)
        fields = self._make_fields(1)
        result = vision.refine_labels(b"fake_png_bytes", fields, 612, 792)

        assert len(result) == 1
        assert result["field_001"] == ("Wages", 0.85)

    def test_refine_labels_invalid_json_string(self):
        """Returns empty dict when LLM returns unparseable string."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "content": "not valid json at all",
        }

        vision = LLMVisionLabelProvider(mock_provider)
        fields = self._make_fields(1)
        result = vision.refine_labels(b"fake_png_bytes", fields, 612, 792)

        assert result == {}

    def test_default_confidence_when_missing(self):
        """Uses 0.8 default confidence when not provided by LLM."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "content": {
                "labels": {"1": "Some field"},
                "confidence": {},  # No confidence for field 1
            },
        }

        vision = LLMVisionLabelProvider(mock_provider)
        fields = self._make_fields(1)
        result = vision.refine_labels(b"fake_png_bytes", fields, 612, 792)

        assert result["field_001"][1] == 0.8


class TestRenderAnnotatedPage:
    """Test page rendering (requires PDF fixtures)."""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a minimal PDF with a form field for testing."""
        from tests.create_test_pdfs import create_fillable_pdf
        pdf_path = tmp_path / "test_form.pdf"
        pdf_path.write_bytes(create_fillable_pdf())
        return pdf_path

    def test_render_returns_png_bytes(self, sample_pdf):
        """render_annotated_page returns valid PNG bytes."""
        field = FormField(
            id="field_001",
            page=1,
            type=FieldType.TEXT,
            position=FieldPosition(x=50, y=700, w=200, h=15),
        )

        img_bytes = render_annotated_page(
            sample_pdf, 0, [field], 612, 792, dpi=72,
        )

        assert isinstance(img_bytes, bytes)
        assert img_bytes[:4] == b"\x89PNG"

    def test_render_no_fields(self, sample_pdf):
        """render_annotated_page works with empty field list."""
        img_bytes = render_annotated_page(
            sample_pdf, 0, [], 612, 792, dpi=72,
        )

        assert isinstance(img_bytes, bytes)
        assert img_bytes[:4] == b"\x89PNG"

    def test_render_field_without_position_skipped(self, sample_pdf):
        """Fields without positions are skipped in rendering."""
        field_no_pos = FormField(
            id="field_002",
            page=1,
            type=FieldType.TEXT,
            position=None,
        )

        img_bytes = render_annotated_page(
            sample_pdf, 0, [field_no_pos], 612, 792, dpi=72,
        )

        assert isinstance(img_bytes, bytes)
        assert img_bytes[:4] == b"\x89PNG"
