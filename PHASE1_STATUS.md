# FormBridge - Phase 1 Complete

## Project Structure

```
/home/admin/clawd/formbridge/
├── pyproject.toml          # Python project config (hatchling, deps)
├── README.md               # Full documentation
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore patterns
├── src/
│   └── formbridge/
│       ├── __init__.py     # Package init, version
│       ├── models.py       # Pydantic models (FormSchema, FormField, etc.)
│       ├── scanner.py      # Scanner module (AcroForm + OCR fallback)
│       └── cli.py          # Click CLI with all commands
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Pytest configuration
│   ├── test_scanner.py     # Scanner tests
│   ├── download_fixtures.py # Downloads IRS forms + creates minimal PDFs
│   ├── create_test_pdfs.py # Creates minimal test PDFs (standalone)
│   └── fixtures/           # Test PDF files (created by download_fixtures.py)
└── verify_structure.py     # Verifies project structure
```

## Installation

```bash
cd /home/admin/clawd/formbridge

# Install the package in development mode
pip install -e ".[dev]"

# With OCR support (optional)
pip install -e ".[dev,ocr]"
```

## Running Tests

```bash
# First, download test fixtures (IRS forms + minimal PDFs)
python tests/download_fixtures.py

# Run all tests
pytest tests/ -v

# Run only unit tests (skip integration)
pytest tests/ -v -m "not integration"
```

## CLI Usage

```bash
# Scan a PDF form
formbridge scan form.pdf --output fields.json

# With table output
formbridge --format table scan form.pdf

# Verbose mode
formbridge -v scan form.pdf

# Check version
formbridge --version

# Help
formbridge --help
formbridge scan --help
```

## Implemented Features

### Scanner Module
- [x] PDF field extraction using pdfplumber (AcroForm native fields)
- [x] OCR fallback for non-fillable PDFs (pytesseract + pdf2image)
- [x] Layout analysis to identify field labels + input areas
- [x] Field type detection (text, checkbox, radio, date, number, signature)
- [x] FormSchema JSON output matching the spec exactly

### CLI
- [x] `formbridge scan <form.pdf>` - Extract fields from PDF
- [x] Global flags: --verbose, --format, --provider, --model
- [x] Stub commands for: parse, fill, verify, template, serve

### Models
- [x] FormSchema - Main form structure
- [x] FormField - Individual field with type, position, constraints
- [x] FieldPosition - x, y, w, h coordinates
- [x] FieldType - text, checkbox, radio, date, number, signature
- [x] InstructionMap, FieldMapping, VerificationReport (for future phases)

### Tests
- [x] Unit tests for models
- [x] Unit tests for scanner logic
- [x] Integration tests with minimal PDFs (always available)
- [x] Integration tests with IRS Form 1065 (optional download)
- [x] CLI tests for all commands

## Not Yet Implemented (Future Phases)

- Parser module (Phase 2)
- Mapper module (Phase 3)
- Writer module (Phase 3)
- Verification workflow (Phase 4)
- Templates (Phase 4)
- MCP server (Phase 5)

## Notes

- The scanner will attempt AcroForm extraction first, then fall back to OCR if no fields found
- OCR requires installing tesseract and poppler system dependencies
- Tests are designed to work with locally-created minimal PDFs (no download required)
- IRS form tests are optional and require running `python tests/download_fixtures.py`
