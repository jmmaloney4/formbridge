"""Visual PDF viewer/editor for FormBridge.

Provides a local web server that renders filled PDFs with interactive field editing.

Usage:
    formbridge view <filled.pdf> [--mapping mapping.json] [--port 8765]

Architecture:
    - Simple HTTP server using Python's built-in http.server (stdlib only)
    - Serves static HTML/JS/CSS frontend
    - JSON API endpoints for PDF, mapping, schema, updates, and saving
    - No external web framework dependencies
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from formbridge.models import (
    FieldMapping,
    FieldMappingResult,
    FieldType,
    FormField,
    FormSchema,
)
from formbridge.scanner import Scanner, ScannerError
from formbridge.writer import PDFWriter, WriterError

logger = logging.getLogger(__name__)

# Path to viewer assets (relative to this file)
VIEWER_ASSETS_DIR = Path(__file__).parent / "viewer_assets"


class ViewerError(Exception):
    """Base error for viewer operations."""
    pass


class ViewerState:
    """Holds the current state of the viewer.

    This includes the PDF bytes, form schema, field mapping,
    and tracks any modifications.
    """

    def __init__(
        self,
        pdf_path: Path,
        schema: FormSchema | None = None,
        mapping_result: FieldMappingResult | None = None,
        original_pdf_path: Path | None = None,
    ) -> None:
        """Initialize viewer state.

        Args:
            pdf_path: Path to the PDF to view/edit
            schema: Optional form schema (if None, will scan PDF)
            mapping_result: Optional field mapping result
            original_pdf_path: Path to original blank PDF (for re-filling)
        """
        self.pdf_path = pdf_path
        self.original_pdf_path = original_pdf_path or pdf_path
        self.schema = schema
        self.mapping_result = mapping_result
        self._pdf_bytes: bytes | None = None
        self._modified = False

        # Load PDF bytes
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        self._pdf_bytes = pdf_path.read_bytes()

    @property
    def pdf_bytes(self) -> bytes:
        """Get current PDF bytes."""
        if self._pdf_bytes is None:
            raise ViewerError("PDF not loaded")
        return self._pdf_bytes

    @pdf_bytes.setter
    def pdf_bytes(self, value: bytes) -> None:
        """Set PDF bytes and mark as modified."""
        self._pdf_bytes = value
        self._modified = True

    @property
    def is_modified(self) -> bool:
        """Check if PDF has been modified."""
        return self._modified

    def get_field_value(self, field_id: str) -> str | None:
        """Get the current value for a field."""
        if not self.mapping_result:
            return None

        # Check regular mappings
        for mapping in self.mapping_result.mappings:
            if mapping.field_id == field_id:
                return mapping.value

        # Check calculations
        for calc in self.mapping_result.calculations:
            if calc.field_id == field_id:
                return calc.value

        return None

    def update_field(self, field_id: str, new_value: str) -> bool:
        """Update a field value and regenerate the PDF.

        Returns:
            True if update was successful
        """
        if not self.mapping_result or not self.schema:
            logger.warning("Cannot update field: no mapping or schema")
            return False

        # Find and update the mapping
        updated = False

        # Check regular mappings
        for mapping in self.mapping_result.mappings:
            if mapping.field_id == field_id:
                mapping.value = new_value
                mapping.confidence = 1.0  # Manual edit = full confidence
                mapping.reasoning = "Manually edited"
                updated = True
                break

        if not updated:
            # Check calculations - override with manual value
            for calc in self.mapping_result.calculations:
                if calc.field_id == field_id:
                    calc.value = new_value
                    calc.confidence = 1.0
                    calc.reasoning = "Manually edited (overriding calculation)"
                    calc.calculated = False
                    updated = True
                    break

        if not updated:
            # Add new mapping for unmapped field
            self.mapping_result.mappings.append(FieldMapping(
                field_id=field_id,
                value=new_value,
                confidence=1.0,
                reasoning="Manually added",
            ))
            updated = True

        if updated:
            # Regenerate PDF
            try:
                self._regenerate_pdf()
            except WriterError as e:
                logger.error(f"Failed to regenerate PDF: {e}")
                return False

        return updated

    def _regenerate_pdf(self) -> None:
        """Regenerate the PDF with current field values."""
        import tempfile

        if not self.schema or not self.mapping_result:
            raise ViewerError("Cannot regenerate PDF: missing schema or mapping")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            writer = PDFWriter(
                pdf_path=self.original_pdf_path,
                form_schema=self.schema,
                verbose=False,
            )
            writer.write(self.mapping_result, tmp_path)

            # Read the new PDF bytes
            self.pdf_bytes = tmp_path.read_bytes()

        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    def save_pdf(self, output_path: Path) -> bool:
        """Save the current PDF to disk."""
        try:
            output_path.write_bytes(self.pdf_bytes)
            self._modified = False
            logger.info(f"Saved PDF to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save PDF: {e}")
            return False


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the viewer.

    Endpoints:
        GET /           → Serves the HTML/JS viewer
        GET /pdf        → Serves the PDF file
        GET /mapping    → Returns the field mapping JSON
        GET /schema     → Returns the form schema JSON
        POST /update    → Update a field value, returns updated PDF
        POST /save      → Save final PDF to disk
    """

    # Class-level state (shared across all requests)
    state: ViewerState | None = None

    def __init__(self, *args, **kwargs) -> None:
        # Set directory for static files
        kwargs["directory"] = str(VIEWER_ASSETS_DIR)
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        """Override to use our logger instead of stderr."""
        logger.debug("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_index()
        elif path == "/pdf":
            self._serve_pdf()
        elif path == "/mapping":
            self._serve_mapping()
        elif path == "/schema":
            self._serve_schema()
        elif path == "/status":
            self._serve_status()
        else:
            # Fall back to static file serving
            super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/update":
            self._handle_update()
        elif path == "/save":
            self._handle_save()
        else:
            self._send_error_response(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_index(self) -> None:
        """Serve the main HTML viewer."""
        index_path = VIEWER_ASSETS_DIR / "index.html"

        if not index_path.exists():
            self._send_error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Viewer assets not found. Check installation."
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        with open(index_path, "rb") as f:
            self.wfile.write(f.read())

    def _serve_pdf(self) -> None:
        """Serve the PDF file."""
        if not self.state:
            self._send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "No state")
            return

        try:
            pdf_bytes = self.state.pdf_bytes

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.send_header("Content-Disposition", f'inline; filename="{self.state.pdf_path.name}"')
            self.end_headers()

            self.wfile.write(pdf_bytes)

        except ViewerError as e:
            self._send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _serve_mapping(self) -> None:
        """Serve the field mapping JSON."""
        if not self.state:
            self._send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "No state")
            return

        if not self.state.mapping_result:
            # Return empty mapping
            data = {"mappings": [], "calculations": [], "warnings": []}
        else:
            data = self.state.mapping_result.model_dump()

        self._send_json_response(data)

    def _serve_schema(self) -> None:
        """Serve the form schema JSON."""
        if not self.state:
            self._send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "No state")
            return

        if not self.state.schema:
            self._send_error_response(HTTPStatus.NOT_FOUND, "No schema available")
            return

        data = self.state.schema.model_dump()
        self._send_json_response(data)

    def _serve_status(self) -> None:
        """Serve viewer status."""
        if not self.state:
            self._send_json_response({"status": "error", "message": "No state"})
            return

        status = {
            "status": "ok",
            "pdf_path": str(self.state.pdf_path),
            "modified": self.state.is_modified,
            "has_schema": self.state.schema is not None,
            "has_mapping": self.state.mapping_result is not None,
            "fields_count": len(self.state.schema.fields) if self.state.schema else 0,
        }

        self._send_json_response(status)

    def _handle_update(self) -> None:
        """Handle field value update."""
        if not self.state:
            self._send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "No state")
            return

        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error_response(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        field_id = data.get("field_id")
        new_value = data.get("value")

        if not field_id:
            self._send_error_response(HTTPStatus.BAD_REQUEST, "Missing field_id")
            return

        # Update the field
        success = self.state.update_field(field_id, new_value)

        if success:
            self._send_json_response({
                "success": True,
                "field_id": field_id,
                "value": new_value,
                "message": "Field updated",
            })
        else:
            self._send_error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Failed to update field"
            )

    def _handle_save(self) -> None:
        """Handle save PDF request."""
        if not self.state:
            self._send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "No state")
            return

        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        # Determine output path
        output_path_str = data.get("path")

        if output_path_str:
            output_path = Path(output_path_str)
        else:
            # Default to same directory with "_edited" suffix
            stem = self.state.pdf_path.stem
            if stem.endswith("_filled"):
                stem = stem[:-7]  # Remove _filled suffix
            output_path = self.state.pdf_path.with_name(f"{stem}_edited.pdf")

        # Save the PDF
        success = self.state.save_pdf(output_path)

        if success:
            self._send_json_response({
                "success": True,
                "path": str(output_path),
                "message": f"Saved to {output_path}",
            })
        else:
            self._send_error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Failed to save PDF"
            )

    def _send_json_response(self, data: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        """Send a JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self.wfile.write(body)

    def _send_error_response(self, status: int, message: str) -> None:
        """Send an error response."""
        self._send_json_response({"error": message, "status": "error"}, status)

    # CORS support for local development
    def end_headers(self) -> None:
        """Add CORS headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handle OPTIONS requests for CORS."""
        self.send_response(HTTPStatus.OK)
        self.end_headers()


class Viewer:
    """Main viewer class that manages the HTTP server."""

    def __init__(
        self,
        pdf_path: str | Path,
        mapping_path: str | Path | None = None,
        schema_path: str | Path | None = None,
        original_pdf_path: str | Path | None = None,
        port: int = 8765,
        open_browser: bool = True,
    ) -> None:
        """Initialize the viewer.

        Args:
            pdf_path: Path to the filled PDF to view
            mapping_path: Optional path to field mapping JSON
            schema_path: Optional path to form schema JSON
            original_pdf_path: Optional path to original blank PDF (for re-filling)
            port: Port to run server on
            open_browser: Whether to auto-open browser
        """
        self.pdf_path = Path(pdf_path)
        self.mapping_path = Path(mapping_path) if mapping_path else None
        self.schema_path = Path(schema_path) if schema_path else None
        self.original_pdf_path = Path(original_pdf_path) if original_pdf_path else None
        self.port = port
        self.open_browser = open_browser

        self._server: HTTPServer | None = None
        self._state: ViewerState | None = None

    def start(self) -> None:
        """Start the viewer server."""
        # Load or scan schema
        schema = self._load_schema()

        # Load mapping if provided
        mapping_result = self._load_mapping()

        # Create state
        self._state = ViewerState(
            pdf_path=self.pdf_path,
            schema=schema,
            mapping_result=mapping_result,
            original_pdf_path=self.original_pdf_path,
        )

        # Set handler state
        ViewerRequestHandler.state = self._state

        # Create server
        self._server = HTTPServer(("localhost", self.port), ViewerRequestHandler)

        url = f"http://localhost:{self.port}"
        print(f"\n  FormBridge Viewer")
        print(f"  {'=' * 40}")
        print(f"  PDF: {self.pdf_path.name}")
        if schema:
            print(f"  Fields: {len(schema.fields)}")
        if mapping_result:
            n_filled = len([m for m in mapping_result.mappings if m.value])
            print(f"  Filled: {n_filled}")
        print(f"  URL: {url}")
        print(f"  Press Ctrl+C to stop\n")

        # Open browser
        if self.open_browser:
            self._open_browser(url)

        # Serve
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopping viewer...")
            self._server.shutdown()

    def _load_schema(self) -> FormSchema | None:
        """Load or scan form schema."""
        if self.schema_path and self.schema_path.exists():
            try:
                data = json.loads(self.schema_path.read_text())
                return FormSchema.model_validate(data)
            except Exception as e:
                logger.warning(f"Failed to load schema: {e}")

        # Try to scan the PDF
        try:
            scanner = Scanner(self.pdf_path)
            return scanner.scan()
        except ScannerError as e:
            logger.warning(f"Failed to scan PDF: {e}")
            return None

    def _load_mapping(self) -> FieldMappingResult | None:
        """Load field mapping from file."""
        if not self.mapping_path or not self.mapping_path.exists():
            return None

        try:
            data = json.loads(self.mapping_path.read_text())
            return FieldMappingResult.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load mapping: {e}")
            return None

    def _open_browser(self, url: str) -> None:
        """Open browser to the viewer URL."""
        import threading
        import time

        def open_after_delay():
            time.sleep(0.5)  # Wait for server to start
            webbrowser.open(url)

        thread = threading.Thread(target=open_after_delay, daemon=True)
        thread.start()


def run_viewer(
    pdf_path: str | Path,
    mapping_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    original_pdf_path: str | Path | None = None,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Convenience function to run the viewer.

    Args:
        pdf_path: Path to the filled PDF to view
        mapping_path: Optional path to field mapping JSON
        schema_path: Optional path to form schema JSON
        original_pdf_path: Optional path to original blank PDF (for re-filling)
        port: Port to run server on
        open_browser: Whether to auto-open browser
    """
    viewer = Viewer(
        pdf_path=pdf_path,
        mapping_path=mapping_path,
        schema_path=schema_path,
        original_pdf_path=original_pdf_path,
        port=port,
        open_browser=open_browser,
    )
    viewer.start()
