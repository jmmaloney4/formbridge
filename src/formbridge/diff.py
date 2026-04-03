"""FormBridge diff - Compare two filled PDF forms field-by-field."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pikepdf

logger = logging.getLogger(__name__)


@dataclass
class FieldDiff:
    """A single field difference between two PDFs."""

    field_name: str
    old_value: str | None
    new_value: str | None
    page: int | None = None
    kind: str = "changed"  # changed | added | removed | unchanged

    @property
    def summary(self) -> str:
        if self.kind == "added":
            return f"+ {self.field_name}: {self.new_value!r}"
        elif self.kind == "removed":
            return f"- {self.field_name}: {self.old_value!r}"
        elif self.kind == "changed":
            return f"~ {self.field_name}: {self.old_value!r} -> {self.new_value!r}"
        else:
            return f"  {self.field_name}: {self.old_value!r}"


@dataclass
class DiffResult:
    """Result of comparing two PDF forms."""

    old_path: str
    new_path: str
    total_fields: int = 0
    added: list[FieldDiff] = field(default_factory=list)
    removed: list[FieldDiff] = field(default_factory=list)
    changed: list[FieldDiff] = field(default_factory=list)
    unchanged: list[FieldDiff] = field(default_factory=list)

    @property
    def has_differences(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total": self.total_fields,
            "changed": len(self.changed),
            "added": len(self.added),
            "removed": len(self.removed),
            "unchanged": len(self.unchanged),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_path": self.old_path,
            "new_path": self.new_path,
            "stats": self.stats,
            "added": [{"field": d.field_name, "value": d.new_value} for d in self.added],
            "removed": [{"field": d.field_name, "value": d.old_value} for d in self.removed],
            "changed": [
                {"field": d.field_name, "old": d.old_value, "new": d.new_value}
                for d in self.changed
            ],
        }


def _extract_field_values(pdf_path: Path) -> dict[str, str | None]:
    """Extract all AcroForm field values from a PDF."""
    values: dict[str, str | None] = {}

    try:
        with pikepdf.open(pdf_path) as pdf:
            if "/AcroForm" not in pdf.Root:
                logger.warning("No AcroForm found in %s", pdf_path)
                return values

            acroform = pdf.Root["/AcroForm"]
            if "/Fields" not in acroform:
                return values

            _walk_fields(acroform["/Fields"], values)
    except Exception as exc:
        logger.error("Failed to read %s: %s", pdf_path, exc)
        raise

    return values


def _walk_fields(
    fields: pikepdf.Array, values: dict[str, str | None], parent_name: str = ""
) -> None:
    """Recursively walk AcroForm field tree and extract values."""
    for field_ref in fields:
        try:
            field_obj = field_ref
            if isinstance(field_ref, pikepdf.Object):
                field_obj = field_ref

            # Get field name
            name = ""
            if "/T" in field_obj:
                name = str(field_obj["/T"])
            full_name = f"{parent_name}.{name}" if parent_name and name else name or parent_name

            # Recurse into child fields
            if "/Kids" in field_obj:
                _walk_fields(field_obj["/Kids"], values, full_name)
                continue

            # Get value
            value = None
            if "/V" in field_obj:
                raw = field_obj["/V"]
                if isinstance(raw, pikepdf.Name):
                    value = str(raw).lstrip("/")
                    # Common checkbox/radio representations
                    if value in ("Off", ""):
                        value = None
                elif isinstance(raw, pikepdf.String):
                    value = str(raw)
                else:
                    value = str(raw)

            if full_name:
                values[full_name] = value

        except Exception as exc:
            logger.debug("Skipping field: %s", exc)
            continue


def diff_pdfs(
    old_path: str | Path,
    new_path: str | Path,
    include_unchanged: bool = False,
) -> DiffResult:
    """Compare two filled PDF forms field-by-field.

    Args:
        old_path: Path to the first (old/baseline) PDF.
        new_path: Path to the second (new/updated) PDF.
        include_unchanged: Whether to include unchanged fields in the result.

    Returns:
        DiffResult with categorized field differences.
    """
    old_path = Path(old_path)
    new_path = Path(new_path)

    if not old_path.exists():
        raise FileNotFoundError(f"Old PDF not found: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New PDF not found: {new_path}")

    old_values = _extract_field_values(old_path)
    new_values = _extract_field_values(new_path)

    all_fields = sorted(set(old_values.keys()) | set(new_values.keys()))

    result = DiffResult(
        old_path=str(old_path),
        new_path=str(new_path),
        total_fields=len(all_fields),
    )

    for field_name in all_fields:
        old_val = old_values.get(field_name)
        new_val = new_values.get(field_name)

        if field_name not in old_values:
            result.added.append(
                FieldDiff(field_name=field_name, old_value=None, new_value=new_val, kind="added")
            )
        elif field_name not in new_values:
            result.removed.append(
                FieldDiff(field_name=field_name, old_value=old_val, new_value=None, kind="removed")
            )
        elif old_val != new_val:
            result.changed.append(
                FieldDiff(field_name=field_name, old_value=old_val, new_value=new_val, kind="changed")
            )
        elif include_unchanged:
            result.unchanged.append(
                FieldDiff(field_name=field_name, old_value=old_val, new_value=new_val, kind="unchanged")
            )

    return result
