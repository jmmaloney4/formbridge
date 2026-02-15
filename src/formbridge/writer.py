"""Writer module for filling PDF forms with mapped data.

This module handles the actual PDF filling:
- Native AcroForm fields using pikepdf (preserves form structure)
- OCR-based fields using reportlab overlay (for non-fillable PDFs)
- Field types: text, checkbox, radio, date
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pikepdf

from formbridge.models import (
    FieldMapping,
    FieldMappingResult,
    FieldType,
    FormField,
    FormSchema,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WriterError(Exception):
    """Base error for writer operations."""
    pass


class PDFFillError(WriterError):
    """Error filling PDF."""
    pass


class FieldValueFormatter:
    """Format values for different field types."""

    @staticmethod
    def format_for_field(
        value: str | None,
        field: FormField,
    ) -> str | None:
        """Format a value for a specific field type.

        Args:
            value: Raw value string
            field: Form field definition

        Returns:
            Formatted value string or None
        """
        if value is None:
            return None

        if field.type == FieldType.DATE:
            return FieldValueFormatter._format_date(value, field)
        elif field.type == FieldType.NUMBER:
            return FieldValueFormatter._format_number(value, field)
        elif field.type == FieldType.CHECKBOX:
            return FieldValueFormatter._format_checkbox(value, field)
        elif field.type == FieldType.TEXT:
            return FieldValueFormatter._format_text(value, field)
        else:
            return str(value)

    @staticmethod
    def _format_date(value: str, field: FormField) -> str:
        """Format date values.

        Accepts various input formats and outputs MM/DD/YYYY (US standard).
        """
        # Common date formats to try
        formats = [
            "%Y-%m-%d",     # ISO format
            "%m/%d/%Y",     # US format
            "%d/%m/%Y",     # European format
            "%Y/%m/%d",     # Alternative ISO
            "%B %d, %Y",    # Month Day, Year
            "%b %d, %Y",    # Abbrev month
            "%d %B %Y",     # Day Month Year
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(value.strip(), fmt)
                return dt.strftime("%m/%d/%Y")
            except ValueError:
                continue

        # If no format matches, return as-is
        return value.strip()

    @staticmethod
    def _format_number(value: str, field: FormField) -> str:
        """Format numeric values.

        Handles commas, decimals, and currency symbols.
        """
        try:
            # Remove formatting
            cleaned = value.replace(",", "").replace("$", "").strip()
            num = float(cleaned)

            # Check if it should be an integer
            if num == int(num):
                return str(int(num))
            else:
                # Round to 2 decimal places
                return f"{num:.2f}"
        except ValueError:
            return value.strip()

    @staticmethod
    def _format_checkbox(value: str, field: FormField) -> str:
        """Format checkbox values.

        Returns the checked value if true-like, empty otherwise.
        """
        # Determine checked value
        checked_value = field.checked_value or "Yes"

        # Interpret value as boolean
        truthy = {
            "true", "yes", "1", "on", "checked", "x", "✓", "☑",
            "selected", "enabled",
        }

        if value.lower().strip() in truthy:
            return checked_value
        else:
            return ""  # Unchecked

    @staticmethod
    def _format_text(value: str, field: FormField) -> str:
        """Format text values.

        Applies max length constraint and basic cleanup.
        """
        text = value.strip()

        # Apply max length if specified
        if field.max_length and len(text) > field.max_length:
            logger.warning(
                f"Text truncated for field {field.id}: "
                f"{len(text)} -> {field.max_length}"
            )
            text = text[:field.max_length]

        return text


class AcroFormFiller:
    """Fill native AcroForm fields using pikepdf."""

    def __init__(self, pdf_path: str | Path) -> None:
        """Initialize with PDF path.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise WriterError(f"PDF file not found: {self.pdf_path}")

        self.pdf: pikepdf.Pdf | None = None

    def open(self) -> None:
        """Open the PDF for editing."""
        try:
            self.pdf = pikepdf.open(self.pdf_path, allow_overwriting_input=True)
        except Exception as e:
            raise WriterError(f"Failed to open PDF: {e}") from e

    def close(self) -> None:
        """Close the PDF."""
        if self.pdf:
            self.pdf.close()
            self.pdf = None

    def __enter__(self) -> AcroFormFiller:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def fill_field(
        self,
        field_name: str,
        value: str | None,
        field_type: FieldType = FieldType.TEXT,
    ) -> bool:
        """Fill a single AcroForm field.

        Args:
            field_name: PDF field name (T attribute)
            value: Value to fill
            field_type: Type of field

        Returns:
            True if field was filled successfully
        """
        if not self.pdf:
            raise WriterError("PDF not open")

        if value is None:
            return False

        try:
            # Get the AcroForm
            if "/AcroForm" not in self.pdf.Root:
                logger.warning("PDF has no AcroForm")
                return False

            acroform = self.pdf.Root.AcroForm

            # Find the field by name
            field = self._find_field_by_name(field_name)
            if field is None:
                logger.warning(f"Field not found: {field_name}")
                return False

            # Set value based on field type
            if field_type == FieldType.CHECKBOX:
                self._set_checkbox_value(field, value)
            elif field_type == FieldType.RADIO:
                self._set_radio_value(field, value)
            else:
                # Text field
                self._set_text_value(field, value)

            return True

        except Exception as e:
            logger.error(f"Failed to fill field {field_name}: {e}")
            return False

    def _find_field_by_name(self, name: str) -> pikepdf.Object | None:
        """Find a field by its name (T attribute).

        Searches recursively through field hierarchy.
        """
        if "/AcroForm" not in self.pdf.Root:
            return None

        acroform = self.pdf.Root.AcroForm
        if "/Fields" not in acroform:
            return None

        return self._search_fields(acroform.Fields, name)

    def _search_fields(
        self,
        fields: pikepdf.Array,
        name: str,
    ) -> pikepdf.Object | None:
        """Recursively search for a field by name."""
        for field in fields:
            # Check this field's name
            if "/T" in field:
                field_name = str(field.T)
                if field_name == name:
                    return field

            # Search child fields
            if "/Kids" in field:
                result = self._search_fields(field.Kids, name)
                if result:
                    return result

        return None

    def _set_text_value(self, field: pikepdf.Object, value: str) -> None:
        """Set a text field value."""
        field.V = value
        # Clear any existing appearance
        if "/AP" in field:
            del field.AP

    def _set_checkbox_value(self, field: pikepdf.Object, value: str) -> None:
        """Set a checkbox field value."""
        if value:  # Checked
            # Use the export value (usually "Yes" or "Off")
            if "/V" in field:
                field.V = value
            # Set appearance
            if "/AS" in field:
                field.AS = value
        else:  # Unchecked
            if "/V" in field:
                field.V = pikepdf.Name("Off")
            if "/AS" in field:
                field.AS = pikepdf.Name("Off")

    def _set_radio_value(self, field: pikepdf.Object, value: str) -> None:
        """Set a radio button field value."""
        # Radio buttons use V to store selected value
        field.V = value

    def save(self, output_path: str | Path) -> None:
        """Save the filled PDF.

        Args:
            output_path: Path to save filled PDF
        """
        if not self.pdf:
            raise WriterError("PDF not open")

        try:
            self.pdf.save(output_path)
            logger.info(f"Saved filled PDF to {output_path}")
        except Exception as e:
            raise WriterError(f"Failed to save PDF: {e}") from e

    def has_acroform(self) -> bool:
        """Check if PDF has AcroForm fields."""
        if not self.pdf:
            self.open()
            close_after = True
        else:
            close_after = False

        try:
            has_form = "/AcroForm" in self.pdf.Root
            if has_form:
                acroform = self.pdf.Root.AcroForm
                return "/Fields" in acroform and len(acroform.Fields) > 0
            return False
        finally:
            if close_after:
                self.close()


class OverlayWriter:
    """Write text overlays on non-fillable PDFs using reportlab."""

    def __init__(self, pdf_path: str | Path) -> None:
        """Initialize with PDF path.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise WriterError(f"PDF file not found: {self.pdf_path}")

        self._overlays: list[dict[str, Any]] = []

    def add_text_overlay(
        self,
        text: str,
        page: int,
        x: float,
        y: float,
        font_size: float = 10,
        font_name: str = "Helvetica",
    ) -> None:
        """Add a text overlay at the specified position.

        Args:
            text: Text to overlay
            page: Page number (1-indexed)
            x: X position (PDF points from left)
            y: Y position (PDF points from bottom)
            font_size: Font size in points
            font_name: Font name
        """
        self._overlays.append({
            "type": "text",
            "text": text,
            "page": page,
            "x": x,
            "y": y,
            "font_size": font_size,
            "font_name": font_name,
        })

    def add_checkbox_overlay(
        self,
        checked: bool,
        page: int,
        x: float,
        y: float,
        size: float = 10,
    ) -> None:
        """Add a checkbox overlay.

        Args:
            checked: Whether checkbox is checked
            page: Page number (1-indexed)
            x: X position (center)
            y: Y position (center)
            size: Size in points
        """
        self._overlays.append({
            "type": "checkbox",
            "checked": checked,
            "page": page,
            "x": x,
            "y": y,
            "size": size,
        })

    def write(self, output_path: str | Path) -> None:
        """Write the PDF with overlays.

        Args:
            output_path: Path to save output PDF
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
        except ImportError as e:
            raise WriterError(
                "reportlab is required for overlay writing. "
                "Install with: pip install reportlab"
            ) from e

        # Group overlays by page
        overlays_by_page: dict[int, list[dict]] = {}
        for overlay in self._overlays:
            page = overlay["page"]
            if page not in overlays_by_page:
                overlays_by_page[page] = []
            overlays_by_page[page].append(overlay)

        # Create overlay PDF
        overlay_buffer = io.BytesIO()
        c = canvas.Canvas(overlay_buffer, pagesize=letter)

        # Get number of pages from original
        import pdfplumber
        with pdfplumber.open(self.pdf_path) as pdf:
            num_pages = len(pdf.pages)
            page_heights = {i + 1: p.height for i, p in enumerate(pdf.pages)}

        for page_num in range(1, num_pages + 1):
            # Get page height for coordinate conversion
            page_height = page_heights.get(page_num, 792)  # Default to letter height

            if page_num in overlays_by_page:
                for overlay in overlays_by_page[page_num]:
                    if overlay["type"] == "text":
                        # PDF y is from bottom, reportlab y is from bottom too
                        # But we need to handle the coordinate system
                        c.setFont(overlay["font_name"], overlay["font_size"])
                        c.drawString(
                            overlay["x"],
                            page_height - overlay["y"],  # Convert from PDF top-origin
                            overlay["text"],
                        )
                    elif overlay["type"] == "checkbox":
                        if overlay["checked"]:
                            # Draw X or checkmark
                            size = overlay["size"]
                            x = overlay["x"]
                            y = page_height - overlay["y"]
                            c.setFont("Helvetica-Bold", size)
                            c.drawString(x, y - size * 0.3, "X")

            c.showPage()

        c.save()
        overlay_buffer.seek(0)

        # Merge overlay with original
        self._merge_overlay(output_path, overlay_buffer)

    def _merge_overlay(
        self,
        output_path: str | Path,
        overlay_buffer: io.BytesIO,
    ) -> None:
        """Merge overlay PDF with original."""
        with pikepdf.open(self.pdf_path) as original:
            with pikepdf.open(overlay_buffer) as overlay:
                # Assuming same number of pages
                for orig_page, overlay_page in zip(original.pages, overlay.pages):
                    # Add overlay content as a new layer
                    orig_page.add_overlay(overlay_page)

                original.save(output_path)


class PDFWriter:
    """Main writer class that coordinates PDF filling."""

    def __init__(
        self,
        pdf_path: str | Path,
        form_schema: FormSchema,
        verbose: bool = False,
    ) -> None:
        """Initialize writer.

        Args:
            pdf_path: Path to original PDF
            form_schema: Form schema with field definitions
            verbose: Enable verbose logging
        """
        self.pdf_path = Path(pdf_path)
        self.form_schema = form_schema
        self.verbose = verbose

        if not self.pdf_path.exists():
            raise WriterError(f"PDF file not found: {self.pdf_path}")

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        # Determine fill mode
        self._has_acroform = self._check_acroform()

    def _check_acroform(self) -> bool:
        """Check if PDF has native AcroForm fields."""
        try:
            with AcroFormFiller(self.pdf_path) as filler:
                return filler.has_acroform()
        except Exception:
            return False

    def write(
        self,
        mapping_result: FieldMappingResult,
        output_path: str | Path,
    ) -> bool:
        """Write the filled PDF.

        Args:
            mapping_result: Field mapping result from mapper
            output_path: Path to save filled PDF

        Returns:
            True if successful
        """
        output_path = Path(output_path)

        # Combine regular mappings and calculations
        all_mappings = list(mapping_result.mappings) + list(mapping_result.calculations)

        # Build field lookup
        field_lookup = {f.id: f for f in self.form_schema.fields}

        if self._has_acroform:
            logger.info("Using AcroForm filling mode")
            return self._write_acroform(all_mappings, field_lookup, output_path)
        else:
            logger.info("Using overlay writing mode")
            return self._write_overlay(all_mappings, field_lookup, output_path)

    def _write_acroform(
        self,
        mappings: list[FieldMapping],
        field_lookup: dict[str, FormField],
        output_path: Path,
    ) -> bool:
        """Fill using AcroForm fields."""
        success_count = 0
        skip_count = 0

        with AcroFormFiller(self.pdf_path) as filler:
            for mapping in mappings:
                if mapping.value is None:
                    skip_count += 1
                    continue

                field = field_lookup.get(mapping.field_id)
                if not field:
                    logger.warning(f"Field not found in schema: {mapping.field_id}")
                    continue

                # Format value for field type
                formatted_value = FieldValueFormatter.format_for_field(
                    mapping.value, field
                )

                if formatted_value is None:
                    skip_count += 1
                    continue

                # Use field.id as field name (this is how scanner names them)
                # But AcroForm fields might have different names - try both
                field_name = self._find_acroform_field_name(filler, field)
                if field_name is None:
                    logger.warning(f"Could not find AcroForm field for {field.id}")
                    continue

                if filler.fill_field(field_name, formatted_value, field.type):
                    success_count += 1
                else:
                    logger.warning(f"Failed to fill field {field.id}")

            filler.save(output_path)

        logger.info(f"Filled {success_count} fields, skipped {skip_count}")
        return success_count > 0

    def _find_acroform_field_name(
        self,
        filler: AcroFormFiller,
        field: FormField,
    ) -> str | None:
        """Find the AcroForm field name for a schema field.

        The scanner generates field IDs like 'field_001', but the PDF
        may have different internal names. This method tries to match.
        """
        # Try exact match first
        if filler._find_field_by_name(field.id):
            return field.id

        # Try label-based match
        if field.label:
            # Normalize label to field name format
            normalized = field.label.replace(" ", "_").replace("-", "_").lower()
            if filler._find_field_by_name(normalized):
                return normalized

        # Try line reference
        if field.line_ref:
            line_names = [
                f"Line{field.line_ref}",
                f"Line_{field.line_ref}",
                f"line{field.line_ref}",
                field.line_ref,
            ]
            for name in line_names:
                if filler._find_field_by_name(name):
                    return name

        # Return field.id as fallback (may not work but gives useful error)
        return field.id

    def _write_overlay(
        self,
        mappings: list[FieldMapping],
        field_lookup: dict[str, FormField],
        output_path: Path,
    ) -> bool:
        """Fill using text overlay (for non-fillable PDFs)."""
        overlay = OverlayWriter(self.pdf_path)

        success_count = 0

        for mapping in mappings:
            if mapping.value is None:
                continue

            field = field_lookup.get(mapping.field_id)
            if not field or not field.position:
                logger.warning(f"Field {mapping.field_id} has no position, skipping")
                continue

            # Format value
            formatted_value = FieldValueFormatter.format_for_field(
                mapping.value, field
            )

            if formatted_value is None:
                continue

            if field.type == FieldType.CHECKBOX:
                # Add checkbox overlay
                overlay.add_checkbox_overlay(
                    checked=bool(formatted_value),
                    page=field.page,
                    x=field.position.x + field.position.w / 2,
                    y=field.position.y + field.position.h / 2,
                    size=min(field.position.w, field.position.h) * 0.6,
                )
            else:
                # Add text overlay
                # Calculate font size based on field height
                font_size = self._calculate_font_size(field)
                overlay.add_text_overlay(
                    text=formatted_value,
                    page=field.page,
                    x=field.position.x + 2,  # Small padding
                    y=field.position.y + field.position.h - 2,  # Top of field
                    font_size=font_size,
                )

            success_count += 1

        if success_count > 0:
            overlay.write(output_path)
            logger.info(f"Created overlay with {success_count} fields")

        return success_count > 0

    def _calculate_font_size(self, field: FormField) -> float:
        """Calculate appropriate font size for field."""
        if not field.position:
            return 10

        # Base size on field height
        height = field.position.h
        font_size = height * 0.7  # 70% of field height

        # Clamp to reasonable range
        return max(6, min(font_size, 14))


def write_filled_pdf(
    pdf_path: str | Path,
    form_schema: FormSchema,
    mapping_result: FieldMappingResult,
    output_path: str | Path,
    verbose: bool = False,
) -> bool:
    """Convenience function to fill a PDF.

    Args:
        pdf_path: Path to original PDF
        form_schema: Form schema
        mapping_result: Field mapping result
        output_path: Path to save filled PDF
        verbose: Enable verbose logging

    Returns:
        True if successful
    """
    writer = PDFWriter(
        pdf_path=pdf_path,
        form_schema=form_schema,
        verbose=verbose,
    )
    return writer.write(mapping_result, output_path)
