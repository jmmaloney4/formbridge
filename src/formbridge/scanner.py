"""Scanner module for extracting fields from PDF forms.

This module provides two approaches to field extraction:
1. Native AcroForm field extraction (preferred for fillable PDFs)
2. OCR-based field detection (fallback for non-fillable PDFs)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pdfplumber

from formbridge.models import (
    FieldPosition,
    FieldType,
    FormField,
    FormSchema,
)

if TYPE_CHECKING:
    from pdfplumber.pdf import PDF

logger = logging.getLogger(__name__)


class ScannerError(Exception):
    """Base error for scanner operations."""

    pass


class PDFFieldExtractor:
    """Extract fields from PDFs with native AcroForm fields."""

    # Mapping of PDF field types to our FieldType enum
    FIELD_TYPE_MAP = {
        "text": FieldType.TEXT,
        "string": FieldType.TEXT,
        "numeric": FieldType.NUMBER,
        "date": FieldType.DATE,
        "checkbox": FieldType.CHECKBOX,
        "radio": FieldType.RADIO,
        "button": FieldType.CHECKBOX,
        "signature": FieldType.SIGNATURE,
    }

    def __init__(self, pdf: PDF) -> None:
        """Initialize with an open PDF."""
        self.pdf = pdf
        self._pages_cache: dict[int, list[dict]] = {}

    def extract_fields(self) -> list[FormField]:
        """Extract all AcroForm fields from the PDF."""
        fields: list[FormField] = []
        field_counter = 0

        # Get form fields from pdfplumber
        # pdfplumber provides access to form fields via .form_fields attribute
        for page_num, page in enumerate(self.pdf.pages, start=1):
            form_fields = self._get_page_form_fields(page, page_num)

            for field_data in form_fields:
                field_counter += 1
                field = self._convert_field(field_data, field_counter, page_num)
                if field:
                    fields.append(field)

        return fields

    def _get_page_form_fields(self, page, page_num: int) -> list[dict]:
        """Get form fields for a specific page.

        pdfplumber stores form fields at the document level, but we can
        determine which page they're on by their bounding box.
        """
        # Cache form fields if not already done
        if not self._pages_cache:
            self._cache_all_form_fields()

        return self._pages_cache.get(page_num, [])

    def _cache_all_form_fields(self) -> None:
        """Cache all form fields grouped by page."""
        if not hasattr(self.pdf, "form_fields") or not self.pdf.form_fields:
            return

        # Get page heights for coordinate conversion
        page_heights = {i + 1: p.height for i, p in enumerate(self.pdf.pages)}

        for field_name, field_data in self.pdf.form_fields.items():
            # Determine which page this field is on
            # pdfplumber fields have a 'page' attribute or we check bounds
            page_num = field_data.get("page_number", 1)

            if page_num not in self._pages_cache:
                self._pages_cache[page_num] = []

            self._pages_cache[page_num].append({
                "name": field_name,
                **field_data
            })

    def _convert_field(
        self, field_data: dict, counter: int, page_num: int
    ) -> FormField | None:
        """Convert a PDF field to our FormField model."""
        field_name = field_data.get("name", f"field_{counter:03d}")

        # Determine field type
        pdf_type = field_data.get("type", "text").lower()
        field_type = self.FIELD_TYPE_MAP.get(pdf_type, FieldType.TEXT)

        # Get position - pdfplumber uses (x0, top, x1, bottom)
        position = None
        if all(k in field_data for k in ["x0", "top", "x1", "bottom"]):
            x0 = field_data["x0"]
            y0 = field_data["bottom"]  # PDF y is from bottom
            x1 = field_data["x1"]
            y1 = field_data["top"]
            position = FieldPosition(
                x=x0,
                y=y0,
                w=x1 - x0,
                h=y1 - y0,
            )

        # Extract label from nearby text
        label = field_data.get("label") or self._extract_label(field_name, field_data)

        # Get max length for text fields
        max_length = field_data.get("max_len")

        # Handle options for radio/checkbox fields
        options = None
        checked_value = None
        if field_type == FieldType.RADIO:
            options = field_data.get("options", [])
        elif field_type == FieldType.CHECKBOX:
            checked_value = field_data.get("value", "Yes")

        # Extract line reference from field name (e.g., "Line1_A" -> "A")
        line_ref = self._extract_line_ref(field_name)

        return FormField(
            id=f"field_{counter:03d}",
            label=label,
            page=page_num,
            type=field_type,
            position=position,
            max_length=max_length,
            required=False,  # PDFs don't always specify this
            line_ref=line_ref,
            options=options,
            checked_value=checked_value,
            label_source="acroform" if label else None,
        )

    def _extract_label(self, field_name: str, field_data: dict) -> str | None:
        """Extract a human-readable label for the field.

        Tries to find nearby text that serves as a label.
        """
        # First check if there's an explicit label/TU entry
        if field_data.get("label"):
            return field_data["label"]

        # Clean up field name as fallback
        name = field_name.replace("_", " ").replace("-", " ")

        # Remove common prefixes
        name = re.sub(r"^(text|field|input|line)\s*", "", name, flags=re.IGNORECASE)

        return name if name else None

    def _extract_line_ref(self, field_name: str) -> str | None:
        """Extract line reference from field name.

        Examples:
            "Line1_A" -> "A"
            "f1_01" -> "1"
            "Line16a" -> "16a"
        """
        # Try to find line reference patterns
        patterns = [
            r"[Ll]ine\d+[_\s]+([a-zA-Z]\w*)$",  # Line1_A -> A (suffix after line number)
            r"[Ll]ine[_\s]*(\d+[a-zA-Z]?)",  # Line1, Line_16a
            r"_(\d+[a-zA-Z]?)_",  # _42_ in middle
            r"_(\d+[a-zA-Z]?)$",  # _1, _16a at end
            r"^(\d+[a-zA-Z]?)_",  # 1_, 16a_ at start
        ]

        for pattern in patterns:
            match = re.search(pattern, field_name)
            if match:
                return match.group(1)

        return None


class OCRFieldDetector:
    """Detect fields in PDFs using OCR (for non-fillable PDFs)."""

    def __init__(self, pdf: PDF) -> None:
        """Initialize with an open PDF."""
        self.pdf = pdf

    def has_ocr_dependencies(self) -> bool:
        """Check if OCR dependencies are available."""
        try:
            import pytesseract  # noqa: F401
            from pdf2image import convert_from_path  # noqa: F401
            return True
        except ImportError:
            return False

    def detect_fields(self) -> list[FormField]:
        """Detect fields using OCR.

        This is a fallback for PDFs without native form fields.
        It looks for:
        - Underlines (common in paper forms)
        - Boxes/rectangles
        - Checkbox markers (□, ☐)
        """
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image
        except ImportError as e:
            raise ScannerError(
                "OCR dependencies not installed. "
                "Install with: pip install formbridge[ocr]"
            ) from e

        fields: list[FormField] = []
        field_counter = 0

        # Convert PDF pages to images
        # Note: We need the file path for pdf2image
        # This is a limitation - we'll work around it
        for page_num, page in enumerate(self.pdf.pages, start=1):
            detected = self._detect_fields_on_page(page, page_num, field_counter)
            field_counter += len(detected)
            fields.extend(detected)

        return fields

    def _detect_fields_on_page(
        self, page, page_num: int, start_counter: int
    ) -> list[FormField]:
        """Detect fields on a single page using layout analysis."""
        fields: list[FormField] = []

        # Get all characters and their positions
        chars = page.chars
        if not chars:
            return fields

        # Find underlines and boxes (potential input areas)
        lines = page.lines
        rects = page.rects

        # Group rectangles that might be input fields
        input_areas = self._identify_input_areas(chars, lines, rects)

        for i, area in enumerate(input_areas, start=1):
            field = FormField(
                id=f"field_{start_counter + i:03d}",
                label=area.get("label"),
                page=page_num,
                type=FieldType.TEXT,
                position=FieldPosition(
                    x=area["x0"],
                    y=area["y0"],
                    w=area["x1"] - area["x0"],
                    h=area["y1"] - area["y0"],
                ),
                required=False,
            )
            fields.append(field)

        # Detect checkboxes
        checkboxes = self._detect_checkboxes(page, page_num, start_counter + len(fields))
        fields.extend(checkboxes)

        return fields

    def _identify_input_areas(
        self, chars: list, lines: list, rects: list
    ) -> list[dict]:
        """Identify potential input areas based on layout.

        Heuristics:
        - Empty rectangles with text above/to the left
        - Underlines with text to the left
        - Groups of underscores (________)
        """
        areas: list[dict] = []

        # Find rectangles that could be input fields
        for rect in rects:
            # Skip very small or very large rectangles
            width = rect["x1"] - rect["x0"]
            height = rect["y1"] - rect["y0"]

            if width < 20 or height < 10:
                continue
            if width > 500 or height > 100:
                continue

            # Check if rectangle is empty or has minimal content
            # Find label text nearby (to the left or above)
            label = self._find_label_nearby(chars, rect)

            areas.append({
                "x0": rect["x0"],
                "y0": rect["y0"],
                "x1": rect["x1"],
                "y1": rect["y1"],
                "label": label,
            })

        # Find underlines (horizontal lines near text)
        for line in lines:
            # Only consider horizontal lines
            if abs(line["y1"] - line["y0"]) > 2:
                continue

            width = abs(line["x1"] - line["x0"])
            if width < 30:  # Skip short lines
                continue

            # Check for text to the left
            label = self._find_label_nearby(chars, {
                "x0": line["x0"] - 100,
                "x1": line["x0"] - 5,
                "y0": min(line["y0"], line["y1"]) - 10,
                "y1": max(line["y0"], line["y1"]) + 10,
            })

            areas.append({
                "x0": line["x0"],
                "y0": min(line["y0"], line["y1"]) - 12,  # Typical text height
                "x1": line["x1"],
                "y1": max(line["y0"], line["y1"]),
                "label": label,
            })

        # Remove overlapping areas
        areas = self._deduplicate_areas(areas)

        return areas

    def _find_label_nearby(self, chars: list, area: dict) -> str | None:
        """Find label text near a given area."""
        nearby_chars = []

        for char in chars:
            # Check if character is to the left or above the area
            if (
                char["x1"] < area["x0"]
                and char["x0"] > area["x0"] - 150  # Within 150 points to the left
                and abs(char["y0"] - area["y0"]) < 15  # Roughly same line
            ):
                nearby_chars.append(char)

        if not nearby_chars:
            return None

        # Sort by x position and concatenate
        nearby_chars.sort(key=lambda c: c["x0"])
        label = "".join(c["text"] for c in nearby_chars).strip()

        # Clean up label
        label = re.sub(r"\s+", " ", label)
        label = label.rstrip(":").strip()

        return label if label else None

    def _detect_checkboxes(
        self, page, page_num: int, start_counter: int
    ) -> list[FormField]:
        """Detect checkbox fields on a page."""
        checkboxes: list[FormField] = []

        # Look for checkbox characters: □, ☐, ◻, or small squares
        checkbox_chars = {"□", "☐", "◻", "▢", "☐"}

        chars = page.chars
        char_positions: dict[tuple[float, float], list] = {}

        # Group characters by approximate position
        for char in chars:
            if char["text"] in checkbox_chars:
                key = (round(char["x0"], 0), round(char["top"], 0))
                if key not in char_positions:
                    char_positions[key] = []
                char_positions[key].append(char)

        for i, (pos, char_list) in enumerate(char_positions.items(), start=1):
            char = char_list[0]
            checkboxes.append(FormField(
                id=f"field_{start_counter + i:03d}",
                label=None,
                page=page_num,
                type=FieldType.CHECKBOX,
                position=FieldPosition(
                    x=char["x0"],
                    y=char["bottom"],
                    w=char["x1"] - char["x0"],
                    h=char["top"] - char["bottom"],
                ),
                required=False,
                checked_value="Yes",
            ))

        return checkboxes

    def _deduplicate_areas(self, areas: list[dict]) -> list[dict]:
        """Remove overlapping or duplicate areas."""
        if len(areas) <= 1:
            return areas

        # Sort by x0, then y0
        areas.sort(key=lambda a: (a["x0"], a["y0"]))

        deduped: list[dict] = []
        for area in areas:
            # Check if this overlaps significantly with any existing area
            is_duplicate = False
            for existing in deduped:
                overlap = self._calculate_overlap(area, existing)
                if overlap > 0.5:  # More than 50% overlap
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append(area)

        return deduped

    def _calculate_overlap(self, a: dict, b: dict) -> float:
        """Calculate overlap ratio between two rectangles."""
        # Calculate intersection
        x_overlap = max(0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
        y_overlap = max(0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))

        if x_overlap == 0 or y_overlap == 0:
            return 0.0

        intersection = x_overlap * y_overlap

        # Calculate union
        a_area = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
        b_area = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
        union = a_area + b_area - intersection

        return intersection / union if union > 0 else 0.0


class Scanner:
    """Main scanner class that coordinates field extraction."""

    def __init__(
        self,
        pdf_path: str | Path,
        verbose: bool = False,
        vision_labels: bool = False,
        llm_provider: Any | None = None,
    ) -> None:
        """Initialize scanner with a PDF file path.

        Args:
            pdf_path: Path to the PDF file
            verbose: Enable verbose logging
            vision_labels: Enable vision-based label refinement (ADR 001)
            llm_provider: Optional LLM provider for vision calls. If not provided
                but vision_labels is True, one will be created from environment config.
        """
        self.pdf_path = Path(pdf_path)
        self.verbose = verbose
        self.vision_labels = vision_labels
        self._llm_provider = llm_provider

        if not self.pdf_path.exists():
            raise ScannerError(f"PDF file not found: {self.pdf_path}")

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    def scan(self) -> FormSchema:
        """Scan the PDF and extract all fields.

        First tries native AcroForm extraction. If no fields found,
        falls back to OCR-based detection.

        Returns:
            FormSchema with all detected fields
        """
        logger.info(f"Scanning PDF: {self.pdf_path}")

        with pdfplumber.open(self.pdf_path) as pdf:
            # Try native field extraction first
            extractor = PDFFieldExtractor(pdf)
            fields = extractor.extract_fields()

            source = "acroform"

            # If no native fields, try OCR
            if not fields:
                logger.info("No AcroForm fields found, trying OCR detection")
                detector = OCRFieldDetector(pdf)

                if detector.has_ocr_dependencies():
                    fields = detector.detect_fields()
                    source = "ocr"
                else:
                    logger.warning(
                        "OCR dependencies not available. "
                        "Install with: pip install formbridge[ocr]"
                    )

            # Create form ID from filename
            form_id = self._generate_form_id()

            # Create schema
            schema = FormSchema(
                form_id=form_id,
                pages=len(pdf.pages),
                fields=fields,
                source_file=self.pdf_path.name,
                created_at=datetime.now(tz=__import__('zoneinfo').ZoneInfo("UTC")).isoformat(),
            )

            logger.info(
                f"Extracted {len(fields)} fields from {len(pdf.pages)} pages "
                f"(source: {source})"
            )

            # Vision label refinement (ADR 001)
            if self.vision_labels and schema.fields:
                self._refine_labels_with_vision(schema)

            return schema

    def _refine_labels_with_vision(self, schema: FormSchema) -> None:
        """Refine field labels using vision LLM on annotated page renders.

        ADR 001: Renders each page with numbered bounding boxes, sends to a
        vision-capable LLM, and updates field labels with the results.
        """
        from formbridge.vision import (
            LLMVisionLabelProvider,
            render_annotated_page,
        )

        # Get or create LLM provider
        provider = self._llm_provider
        if provider is None:
            from formbridge.llm import create_provider
            provider = create_provider()
            self._llm_provider = provider

        vision = LLMVisionLabelProvider(provider)

        # Group fields by page
        pages: dict[int, list[FormField]] = {}
        for field in schema.fields:
            pages.setdefault(field.page, []).append(field)

        # Get page dimensions
        with pdfplumber.open(self.pdf_path) as pdf:
            page_dims = {
                i + 1: (p.width, p.height)
                for i, p in enumerate(pdf.pages)
            }

        for page_num, page_fields in sorted(pages.items()):
            # Filter to fields with positions
            positioned = [f for f in page_fields if f.position]
            if not positioned:
                continue

            dims = page_dims.get(page_num)
            if not dims:
                continue
            pw, ph = dims

            logger.info(
                f"Vision refinement: rendering page {page_num} "
                f"with {len(positioned)} fields"
            )

            try:
                img_bytes = render_annotated_page(
                    self.pdf_path,
                    page_num - 1,  # 0-based index
                    positioned,
                    pw, ph,
                )
            except Exception as e:
                logger.error(f"Failed to render page {page_num}: {e}")
                continue

            try:
                labels = vision.refine_labels(img_bytes, positioned, pw, ph)
            except Exception as e:
                logger.error(f"Vision label call failed for page {page_num}: {e}")
                continue

            # Update field labels
            for field in positioned:
                if field.id in labels:
                    label_text, confidence = labels[field.id]
                    field.label = label_text
                    field.label_source = "vision"
                    field.label_confidence = confidence

    def _generate_form_id(self) -> str:
        """Generate a form ID from the filename."""
        # Clean up filename to create ID
        name = self.pdf_path.stem
        # Replace spaces and special chars
        name = re.sub(r"[^\w\-]", "-", name)
        name = re.sub(r"-+", "-", name)
        name = name.strip("-").lower()

        return name

    def to_json(self, schema: FormSchema | None = None) -> str:
        """Export schema as JSON string.

        Args:
            schema: Schema to export (if None, scans first)

        Returns:
            JSON string of the schema
        """
        if schema is None:
            schema = self.scan()

        return schema.model_dump_json(indent=2)


def scan_pdf(
    pdf_path: str | Path,
    verbose: bool = False,
    vision_labels: bool = False,
) -> FormSchema:
    """Convenience function to scan a PDF.

    Args:
        pdf_path: Path to the PDF file
        verbose: Enable verbose logging
        vision_labels: Enable vision-based label refinement (ADR 001)

    Returns:
        FormSchema with all detected fields
    """
    scanner = Scanner(pdf_path, verbose=verbose, vision_labels=vision_labels)
    return scanner.scan()
