"""Tests for the formbridge.diff module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from formbridge.diff import DiffResult, FieldDiff, diff_pdfs


class TestFieldDiff:
    def test_changed_summary(self):
        d = FieldDiff(field_name="name", old_value="Alice", new_value="Bob", kind="changed")
        assert "~" in d.summary
        assert "Alice" in d.summary
        assert "Bob" in d.summary

    def test_added_summary(self):
        d = FieldDiff(field_name="email", old_value=None, new_value="a@b.com", kind="added")
        assert "+" in d.summary

    def test_removed_summary(self):
        d = FieldDiff(field_name="phone", old_value="123", new_value=None, kind="removed")
        assert "-" in d.summary

    def test_unchanged_summary(self):
        d = FieldDiff(field_name="ssn", old_value="XXX", new_value="XXX", kind="unchanged")
        assert "ssn" in d.summary


class TestDiffResult:
    def test_has_differences_true(self):
        r = DiffResult(old_path="a.pdf", new_path="b.pdf")
        r.changed.append(FieldDiff(field_name="x", old_value="1", new_value="2", kind="changed"))
        assert r.has_differences is True

    def test_has_differences_false(self):
        r = DiffResult(old_path="a.pdf", new_path="b.pdf")
        assert r.has_differences is False

    def test_stats(self):
        r = DiffResult(old_path="a.pdf", new_path="b.pdf", total_fields=5)
        r.added.append(FieldDiff(field_name="a", old_value=None, new_value="1", kind="added"))
        r.changed.append(FieldDiff(field_name="b", old_value="1", new_value="2", kind="changed"))
        stats = r.stats
        assert stats["total"] == 5
        assert stats["added"] == 1
        assert stats["changed"] == 1
        assert stats["removed"] == 0

    def test_to_dict(self):
        r = DiffResult(old_path="a.pdf", new_path="b.pdf", total_fields=2)
        r.changed.append(FieldDiff(field_name="f1", old_value="old", new_value="new", kind="changed"))
        d = r.to_dict()
        assert d["old_path"] == "a.pdf"
        assert len(d["changed"]) == 1
        assert d["changed"][0]["old"] == "old"


class TestDiffPdfs:
    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            diff_pdfs(tmp_path / "missing.pdf", tmp_path / "also_missing.pdf")

    @patch("formbridge.diff._extract_field_values")
    def test_identical_forms(self, mock_extract: MagicMock, tmp_path: Path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF-1.4 dummy")
        b.write_bytes(b"%PDF-1.4 dummy")

        mock_extract.side_effect = [
            {"name": "Alice", "ssn": "123"},
            {"name": "Alice", "ssn": "123"},
        ]
        result = diff_pdfs(a, b)
        assert not result.has_differences
        assert result.stats["total"] == 2

    @patch("formbridge.diff._extract_field_values")
    def test_changed_field(self, mock_extract: MagicMock, tmp_path: Path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF-1.4 dummy")
        b.write_bytes(b"%PDF-1.4 dummy")

        mock_extract.side_effect = [
            {"name": "Alice", "amount": "100"},
            {"name": "Bob", "amount": "100"},
        ]
        result = diff_pdfs(a, b)
        assert len(result.changed) == 1
        assert result.changed[0].field_name == "name"
        assert result.changed[0].old_value == "Alice"
        assert result.changed[0].new_value == "Bob"

    @patch("formbridge.diff._extract_field_values")
    def test_added_and_removed(self, mock_extract: MagicMock, tmp_path: Path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF-1.4 dummy")
        b.write_bytes(b"%PDF-1.4 dummy")

        mock_extract.side_effect = [
            {"old_field": "val"},
            {"new_field": "val2"},
        ]
        result = diff_pdfs(a, b)
        assert len(result.added) == 1
        assert len(result.removed) == 1
        assert result.added[0].field_name == "new_field"
        assert result.removed[0].field_name == "old_field"

    @patch("formbridge.diff._extract_field_values")
    def test_include_unchanged(self, mock_extract: MagicMock, tmp_path: Path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF-1.4 dummy")
        b.write_bytes(b"%PDF-1.4 dummy")

        mock_extract.side_effect = [
            {"same": "val", "diff": "old"},
            {"same": "val", "diff": "new"},
        ]
        result = diff_pdfs(a, b, include_unchanged=True)
        assert len(result.unchanged) == 1
        assert result.unchanged[0].field_name == "same"
