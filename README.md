# FormBridge

**Instruction-aware PDF form filling for LLMs and humans.**

[![PyPI version](https://img.shields.io/pypi/v/formbridge.svg)](https://pypi.org/project/formbridge/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/nilsyai/formbridge/actions/workflows/test.yml/badge.svg)](https://github.com/nilsyai/formbridge/actions/workflows/test.yml)

FormBridge is an open-source CLI tool that fills any PDF form accurately by combining OCR, official instructions ingestion, and AI-powered field mapping. It's the only tool that reads the actual form instructions to determine what goes in each field.

## Why FormBridge?

**The problem:** Filling PDF forms programmatically is error-prone. You need to know exactly which field gets which data, and a single mistake on forms like IRS 5472 can mean a $25,000 penalty.

**Existing solutions:**
- **TurboTax, H&R Block**: Proprietary, expensive, tax-only
- **Adobe Acrobat**: Manual entry, no intelligence
- **pdf-lib, pikepdf**: You figure out what goes where
- **ChatGPT/Claude**: Can't see the form, hallucinates positions

**FormBridge's approach:** Read the actual form instructions, use LLMs to understand what data goes in each field, and give you confidence scores so you know what to verify.

## Key Features

- **Instruction-aware filling** - Reads official instructions to determine field mappings
- **Confidence scoring** - Every mapping has a 0-1 confidence score with reasoning
- **Form-agnostic** - Works with IRS forms, immigration forms, insurance claims, any PDF
- **Template system** - Scan once, fill many times
- **MCP server** - Works with Claude Desktop, OpenClaw, any MCP client
- **Verification workflow** - Interactive review before writing
- **PDF diff** - Compare two filled forms field-by-field to see what changed

## Quick Install

```bash
pip install formbridge

# With OCR support for non-fillable PDFs
pip install formbridge[ocr]

# With MCP server support
pip install formbridge[mcp]

# Everything
pip install formbridge[all]
```

## Quick Start

### 1. Fill a form with a template

```bash
# Install a template from the registry
formbridge template install irs-1065-2025

# Generate a data template
formbridge data-template irs-1065-2025 --output my-data.json

# Edit my-data.json with your information, then fill
formbridge fill irs-1065-2025 --data my-data.json --output filled-1065.pdf --verify
```

### 2. Fill a form without a template

```bash
# One-shot fill with instructions
formbridge fill form.pdf \
  --data my-data.json \
  --instructions instructions.pdf \
  --output filled.pdf \
  --verify
```

### 3. Scan and understand a form

```bash
# Extract field structure
formbridge scan form.pdf --output fields.json

# Parse instructions
formbridge parse instructions.pdf --fields fields.json --output instructions.json
```

## CLI Reference

### `formbridge scan`

Scan a PDF form and extract field structure.

```bash
formbridge scan <form.pdf> [--output fields.json]
```

Outputs a `FormSchema` with all detected fields, types, positions, and labels.

### `formbridge parse`

Parse instruction documents and map to fields.

```bash
formbridge parse <instructions.pdf> --fields fields.json [--output instructions.json]
```

Requires a scanned form schema. Extracts per-field guidance from instruction documents.

### `formbridge fill`

Fill a PDF form with data.

```bash
formbridge fill <form-or-template> --data data.json --output filled.pdf [--verify] [--dry-run]
```

Options:
- `--verify` - Interactive verification before writing
- `--dry-run` - Show mapping without writing PDF
- `--instructions <path>` - Additional instruction document

### `formbridge verify`

Verify a filled form.

```bash
formbridge verify <filled.pdf> --template <template-name>
```

Shows field-by-field breakdown of what was filled.

### `formbridge diff`

Compare two filled PDFs field-by-field.

```bash
formbridge diff <old.pdf> <new.pdf>            # Show changes
formbridge diff old.pdf new.pdf --all          # Include unchanged fields
formbridge diff old.pdf new.pdf --json         # JSON output
formbridge diff old.pdf new.pdf -r report.json # Save diff report
```

Shows added, removed, and changed fields with color-coded output.
Useful for reviewing iterative edits to tax forms, immigration applications, or any PDF workflow.

### `formbridge template`

Manage templates.

```bash
formbridge template list                    # List installed templates
formbridge template create <form> [inst]    # Create new template
formbridge template install <name>          # Install from registry
formbridge template get <name>              # Show template details
formbridge template delete <name>           # Delete template
```

### `formbridge serve`

Start the MCP server.

```bash
formbridge serve --stdio     # For Claude Desktop, OpenClaw
formbridge serve --port 3000 # HTTP/SSE mode
```

### `formbridge data-template`

Generate a blank data template for a form.

```bash
formbridge data-template <template-name> --output data.json
```

## MCP Server Setup

FormBridge ships as an MCP server so any AI agent can use it as a tool.

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "formbridge": {
      "command": "formbridge",
      "args": ["serve", "--stdio"],
      "env": {
        "FORMBRIDGE_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### OpenClaw

Add to your `openclaw.json`:

```json
{
  "mcpServers": {
    "formbridge": {
      "command": "formbridge",
      "args": ["serve", "--stdio"],
      "env": {
        "FORMBRIDGE_PROVIDER": "openai",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `formbridge_scan` | Scan a PDF form, extract field structure |
| `formbridge_fill` | Fill a form with data using instruction-aware mapping |
| `formbridge_verify` | Verify a filled form, get field breakdown |
| `formbridge_templates` | List available templates |
| `formbridge_template_create` | Create a template from form + instructions |

## Template System

Templates bundle a scanned form + parsed instructions into a reusable package.

### Structure

```
~/.formbridge/templates/irs-1065-2025/
├── manifest.json          # Metadata
├── form.pdf               # Original blank form
├── schema.json            # Scanner output
├── instructions.json      # Parser output (optional)
└── instructions-source.pdf # Original instructions (optional)
```

### Creating Templates

```bash
formbridge template create \
  f1065.pdf \
  i1065-instructions.pdf \
  --name irs-1065-2025 \
  --display-name "IRS Form 1065 (2025)" \
  --category tax/us/irs \
  --tag irs --tag partnership --tag 2025
```

### Template Registry

Community templates are available at [github.com/nilsyai/formbridge-templates](https://github.com/nilsyai/formbridge-templates).

```bash
formbridge template install irs-1065-2025
formbridge template install uscis-i130-2025
```

## LLM Provider Configuration

FormBridge supports multiple LLM providers via environment variables or `formbridge.toml`.

### Environment Variables

```bash
export FORMBRIDGE_PROVIDER=openai
export FORMBRIDGE_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...

# Or for Anthropic
export FORMBRIDGE_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Or for local models (Ollama)
export FORMBRIDGE_PROVIDER=local
export FORMBRIDGE_API_BASE=http://localhost:11434/v1
export FORMBRIDGE_MODEL=llama3.1:8b
```

### Configuration File

Copy `formbridge.toml.example` to `formbridge.toml`:

```toml
[llm]
provider = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"

[llm.local]
base_url = "http://localhost:11434/v1"
model = "llama3.1:8b"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FormBridge CLI                    │
├──────────┬──────────┬───────────┬───────────────────┤
│  Scanner │  Parser  │  Mapper   │  Writer           │
│  (OCR)   │  (Inst.) │  (LLM)    │  (PDF Fill)       │
├──────────┴──────────┴───────────┴───────────────────┤
│              Form Template System                    │
│         (.formbridge/ packages - reusable)           │
├─────────────────────────────────────────────────────┤
│              LLM Provider Abstraction                │
│     (OpenAI / Anthropic / Local / Any compatible)    │
├─────────────────────────────────────────────────────┤
│              MCP Server (optional)                   │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
USER DATA (JSON)      BLANK FORM (PDF)      INSTRUCTIONS (PDF)
       │                     │                      │
       └──────────┬──────────┴──────────────────────┘
                  │
           ┌──────▼──────┐
           │   Scanner   │ → FormSchema
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │   Parser    │ → InstructionMap
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │   Mapper    │ → FieldMapping (with confidence)
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │   Writer    │ → Filled PDF
           └─────────────┘
```

## Verification Workflow

Every fill produces a verification report:

```
FormBridge Fill Verification
Form: IRS Form 1065 (2025)
Fields: 45 mapped, 44 blank, 1 flagged

Page 1:
  ✅ Line A: Name ...................... "Acme Partners LLC" (conf: 0.99)
  ✅ Line B: EIN ....................... "82-1234567" (conf: 0.99)
  ⚠️  Line D: Date started .............. "03/15/2023" (conf: 0.78)
     └─ NOTE: Verify this is formation date, not filing date

Overall confidence: 0.96
```

Confidence thresholds:
- **≥ 0.95**: Auto-fill, no review needed
- **0.80 - 0.94**: Fill but flag for review
- **< 0.80**: Leave blank, requires manual input

## Python API

```python
from formbridge import Scanner, Mapper, PDFWriter, load_llm_config

# Scan a form
scanner = Scanner("form.pdf")
schema = scanner.scan()

# Load LLM config
llm_config = load_llm_config()

# Map data to fields
mapper = Mapper(
    user_data={"name": "Acme Corp", "ein": "12-3456789"},
    form_schema=schema,
    llm_config=llm_config,
)
mapping = mapper.map()

# Write filled PDF
writer = PDFWriter("form.pdf", schema)
writer.write(mapping, "filled.pdf")
```

## Requirements

- Python 3.11+
- For OCR: Tesseract (install via system package manager)

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

## Contributing

Contributions welcome! Areas of interest:

- New templates for common forms
- OCR improvements for non-fillable PDFs
- Additional LLM provider support
- Visual verification (rendered previews)
- Batch filling workflows

1. Fork the repo
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF text extraction
- [pikepdf](https://github.com/pikepdf/pikepdf) - PDF manipulation
- [Pydantic](https://github.com/pydantic/pydantic) - Data validation
- [MCP](https://modelcontextprotocol.io) - Model Context Protocol

---

Made with ❤️ by [Nilsy](https://github.com/nilsyai)
