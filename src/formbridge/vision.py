"""Vision-augmented label extraction for AcroForm fields (ADR 001).

Renders PDF pages to annotated images with numbered bounding boxes,
sends them to a vision-capable LLM, and maps the responses back to
field labels.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Protocol

from formbridge.models import FormField

logger = logging.getLogger(__name__)


class VisionLabelProvider(Protocol):
    """Protocol for vision label refinement backends."""

    def refine_labels(
        self,
        page_image: bytes,
        fields: list[FormField],
        page_width: float,
        page_height: float,
    ) -> dict[str, tuple[str, float]]:
        """Return a mapping of field_id -> (label, confidence).

        Args:
            page_image: PNG bytes of the annotated page render.
            fields: Fields on this page with positions.
            page_width: Page width in PDF points.
            page_height: Page height in PDF points.

        Returns:
            Dict mapping field_id to (label_text, confidence_0_to_1).
        """
        ...


def pdf_to_pixel(
    pdf_x: float,
    pdf_y: float,
    pdf_width: float,
    pdf_height: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float]:
    """Convert PDF coordinates (points, origin bottom-left) to pixel coords (origin top-left).

    ADR 001 section: Coordinate conversion.
    """
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height
    px = pdf_x * scale_x
    py = (pdf_height - pdf_y) * scale_y  # flip Y axis
    return px, py


def render_annotated_page(
    pdf_path: Any,
    page_index: int,
    fields: list[FormField],
    page_width: float,
    page_height: float,
    dpi: int = 200,
) -> bytes:
    """Render a PDF page with numbered bounding boxes overlaid on each field.

    Args:
        pdf_path: Path to the PDF file.
        page_index: 0-based page index.
        fields: Fields on this page (must have .position set).
        page_width: Page width in PDF points.
        page_height: Page height in PDF points.
        dpi: Render resolution.

    Returns:
        PNG bytes of the annotated page image.
    """
    import pdfplumber
    from PIL import Image, ImageDraw, ImageFont

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        img = page.to_image(resolution=dpi)
        pil_img = img.original  # PIL Image

    draw = ImageDraw.Draw(pil_img)
    img_w, img_h = pil_img.size

    # Try to get a small font for labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default(size=10)
        except TypeError:
            font = ImageFont.load_default()

    for i, field in enumerate(fields, start=1):
        if not field.position:
            continue

        # PDF points -> pixel coordinates
        x1_px, y2_px = pdf_to_pixel(
            field.position.x,
            field.position.y,
            page_width, page_height,
            img_w, img_h,
        )
        x2_px, y1_px = pdf_to_pixel(
            field.position.x + field.position.w,
            field.position.y + field.position.h,
            page_width, page_height,
            img_w, img_h,
        )

        # Draw numbered rectangle
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=2)
        draw.text((x1_px, y1_px - 12), str(i), fill="red", font=font)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def build_field_list_text(fields: list[FormField]) -> str:
    """Build the field list text for the vision prompt."""
    lines = []
    for i, field in enumerate(fields, start=1):
        pos = ""
        if field.position:
            pos = f" at ({field.position.x:.1f}, {field.position.y:.1f})"
        existing_label = f" (current: {field.label})" if field.label else ""
        lines.append(f"- [{i}] {field.id} ({field.type.value}){pos}{existing_label}")
    return "\n".join(lines)


def build_vision_prompt(fields: list[FormField]) -> str:
    """Build the prompt sent to the vision LLM."""
    return f"""You are looking at a PDF form with numbered red rectangles marking fillable fields.
For each numbered field, identify the printed label or description on the form.

Return a JSON object mapping field numbers to their labels.
Example: {{"1": "First name and middle initial", "2": "Last name", "3": "Your social security number"}}

If a field has no visible label (e.g., it's a calculation field or continuation area),
use the line number or a short description of what appears nearest to it.

Fields on this page:
{build_field_list_text(fields)}

Return ONLY the JSON object, no other text."""


class LLMVisionLabelProvider:
    """Vision label provider that uses an LLM provider with multimodal support.

    Uses the LiteLLMProvider from llm.py to make vision calls.
    """

    def __init__(self, llm_provider: Any) -> None:
        """Initialize with an LLM provider instance.

        Args:
            llm_provider: Any object implementing the LLMProvider protocol
                (must support messages with image_url content blocks).
        """
        self._provider = llm_provider

    def refine_labels(
        self,
        page_image: bytes,
        fields: list[FormField],
        page_width: float,
        page_height: float,
    ) -> dict[str, tuple[str, float]]:
        """Call vision LLM to identify field labels from an annotated page image."""
        import base64

        img_b64 = base64.b64encode(page_image).decode("utf-8")
        prompt = build_vision_prompt(fields)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }
        ]

        # Schema for structured output
        response_schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Mapping of field number (as string) to label text",
                },
                "confidence": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Mapping of field number (as string) to confidence score (0.0-1.0)",
                },
            },
            "required": ["labels", "confidence"],
        }

        try:
            response = self._provider.complete(
                messages=messages,
                schema=response_schema,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"Vision LLM call failed: {e}")
            return {}

        content = response.get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse vision response as JSON: {content}")
                return {}

        labels_raw = content.get("labels", {})
        confidence_raw = content.get("confidence", {})

        result: dict[str, tuple[str, float]] = {}
        for i, field in enumerate(fields, start=1):
            num_str = str(i)
            if num_str in labels_raw:
                label_text = labels_raw[num_str]
                conf = confidence_raw.get(num_str, 0.8)
                result[field.id] = (label_text, float(conf))

        logger.info(
            f"Vision refinement: matched {len(result)}/{len(fields)} fields"
        )
        return result
