# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-03

### Added

- **Diff Command** (`formbridge diff`)
  - Compare two filled PDFs field-by-field
  - Shows added, removed, and changed fields with old/new values
  - Rich CLI output with color-coded tables
  - JSON output mode (`--json`) for programmatic use
  - Save diff reports to file (`--report`)
  - Include unchanged fields with `--all`
  - Useful for reviewing iterative edits to tax forms, immigration applications, etc.

- **diff module** (`formbridge.diff`)
  - `diff_pdfs()` function for programmatic PDF form comparison
  - `DiffResult` and `FieldDiff` dataclasses with serialization
  - AcroForm field value extraction via pikepdf
  - Recursive field tree walking for nested form structures
  - Comprehensive test suite with mocked PDF extraction

## [0.1.0] - 2026-02-15

### Added

- **Core Modules**
  - Scanner: Extract fields from PDF forms (AcroForm native + OCR fallback)
  - Parser: Parse instruction documents and map to field IDs
  - Mapper: AI-powered data-to-field mapping with confidence scoring
  - Writer: Fill PDFs with mapped values (AcroForm + overlay modes)

- **CLI Commands**
  - `formbridge scan`: Scan a PDF and extract field structure
  - `formbridge parse`: Parse instruction documents
  - `formbridge fill`: Fill a form with data using instruction-aware mapping
  - `formbridge verify`: Verify a filled form
  - `formbridge data-template`: Generate a blank data template
  - `formbridge serve`: Start MCP server

- **Template System**
  - Create, list, get, delete templates
  - Install templates from GitHub registry
  - Templates bundle scanned forms + parsed instructions for reuse

- **MCP Server**
  - stdio and HTTP/SSE transports
  - Tools: `formbridge_scan`, `formbridge_fill`, `formbridge_verify`, `formbridge_templates`, `formbridge_template_create`

- **LLM Integration**
  - OpenAI-compatible provider (OpenAI, Ollama, LM Studio, etc.)
  - Anthropic Claude provider
  - Configurable via environment variables or `formbridge.toml`

- **Verification Workflow**
  - Confidence scoring (0.0-1.0) for each field mapping
  - Interactive verification mode with rich CLI output
  - Dry-run mode for programmatic use
  - Verification reports in JSON format

### Technical Details

- Python 3.11+ support
- Built with pdfplumber, pikepdf, reportlab, Pydantic
- Full type hints with mypy strict mode
- Comprehensive test suite (214+ tests)

### Acknowledgments

- Built by Nilsy (nilsyai)
- Inspired by the need for accurate, instruction-aware PDF form filling
- Thanks to all open-source libraries that made this possible

[0.1.0]: https://github.com/nilsyai/formbridge/releases/tag/v0.1.0
