"""FormBridge - Instruction-aware PDF form filling.

An open-source CLI tool that enables LLMs to accurately fill any PDF form
by combining OCR, official instructions ingestion, and AI-powered field mapping.

Example usage:
    >>> from formbridge import Scanner, Mapper, PDFWriter
    >>> from formbridge.models import FormSchema
    >>> 
    >>> # Scan a form
    >>> scanner = Scanner("form.pdf")
    >>> schema = scanner.scan()
    >>> 
    >>> # Map data to fields
    >>> mapper = Mapper(data, schema)
    >>> mapping = mapper.map()
    >>> 
    >>> # Write filled PDF
    >>> writer = PDFWriter("form.pdf", schema)
    >>> writer.write(mapping, "filled.pdf")
"""

from formbridge.models import (
    CalculationResult,
    CalculationRule,
    FieldInstruction,
    FieldMapping,
    FieldMappingEntry,
    FieldMappingResult,
    FieldPosition,
    FieldType,
    FormField,
    FormSchema,
    InstructionMap,
    MappingWarning,
    VerificationReport,
)
from formbridge.scanner import Scanner, ScannerError, scan_pdf
from formbridge.parser import Parser, ParserError, parse_instructions
from formbridge.mapper import Mapper, MapperError, map_data_to_fields
from formbridge.writer import PDFWriter, WriterError, write_filled_pdf
from formbridge.templates import (
    Template,
    TemplateError,
    TemplateManager,
    TemplateManifest,
    TemplateNotFoundError,
    create_template,
    get_template,
    list_templates,
)
from formbridge.llm import (
    LLMConfig,
    LLMError,
    LLMProvider,
    create_provider,
    load_config as load_llm_config,
)

__version__ = "0.1.0"
__author__ = "Nilsy"
__email__ = "nilsyai@users.noreply.github.com"
__license__ = "MIT"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Models
    "CalculationResult",
    "CalculationRule",
    "FieldInstruction",
    "FieldMapping",
    "FieldMappingEntry",
    "FieldMappingResult",
    "FieldPosition",
    "FieldType",
    "FormField",
    "FormSchema",
    "InstructionMap",
    "MappingWarning",
    "VerificationReport",
    # Scanner
    "Scanner",
    "ScannerError",
    "scan_pdf",
    # Parser
    "Parser",
    "ParserError",
    "parse_instructions",
    # Mapper
    "Mapper",
    "MapperError",
    "map_data_to_fields",
    # Writer
    "PDFWriter",
    "WriterError",
    "write_filled_pdf",
    # Templates
    "Template",
    "TemplateError",
    "TemplateManager",
    "TemplateManifest",
    "TemplateNotFoundError",
    "create_template",
    "get_template",
    "list_templates",
    # LLM
    "LLMConfig",
    "LLMError",
    "LLMProvider",
    "create_provider",
    "load_llm_config",
]
