# ADR 001: Vision-Augmented Label Extraction for AcroForm Fields

*Date:* 2026-04-05
*Status:* accepted

## Context

FormBridge's scanner discovers AcroForm fields via `pdfplumber` and produces a `FormSchema` with field positions, types, and labels. The label extraction in `PDFFieldExtractor._extract_label()` (scanner.py:162) is limited to:

1. Checking for a `label` attribute from pdfplumber's `form_fields` dict
2. Falling back to cleaning up the raw field name (e.g., `f1_01[0]` -> `f1 01[0]`)

For IRS tax forms (Form 1040, Schedule D, etc.) this produces unusable results:

- pdfplumber returns the internal AcroForm names (`f1_01[0]`, `c1_01[0]`), not semantic labels
- The `_extract_label()` fallback just strips underscores -- it produces `"f1 01[0]"`, not "First name and middle initial"
- On a full scan of Form 1040, 205 of 263 fields had non-null labels but all were garbled text fragments like `"Aif trteaqcuhi rSecdh.. B"` and `"dsah"`, while 58 had `null` labels

This matters because the mapper (`mapper.py`) builds LLM prompts from field labels to match user data to fields. When labels are opaque internal names or garbage, the LLM cannot reliably map `"wages"` -> `field_042`.

Meanwhile, vision-based OCR (render PDF page to image, send to a vision LLM) reliably reads what is printed on the form. In testing, `pdftoppm` + `vision_analyze` successfully extracted K-1 line values that no text extraction tool could read.

The key insight: the scanner already has precise field positions (bounding boxes in PDF points). Vision can read what a human sees. Combining the two -- annotated page renders + positional field data -- should produce accurate semantic labels.

## Decision

Add a vision-based label refinement pass to the scanner pipeline that runs after the initial AcroForm field extraction.

The refinement works at page granularity, not per-field:

1. After `PDFFieldExtractor.extract_fields()` discovers fields with their bounding boxes, group fields by page
2. For each page, render the full page to a high-resolution PNG
3. Annotate the image with numbered bounding boxes marking each field position
4. Send the annotated image to a vision-capable LLM with a structured prompt asking it to identify the printed label for each numbered field
5. Update `FormField.label` with the vision-derived text

This is a separate refinement step, not a replacement of the existing extraction. The scanner pipeline becomes:

```
scan()
  -> PDFFieldExtractor.extract_fields()   (existing: discovers fields + positions)
  -> _refine_labels_with_vision()          (new: batch vision pass per page)
  -> return FormSchema
```

### Model changes

Add label provenance fields to `FormField`:

```python
class FormField(BaseModel):
    # ... existing fields ...
    label_source: str | None = Field(
        default=None,
        description="How the label was derived: 'acroform', 'vision', 'ocr', or None"
    )
    label_confidence: float | None = Field(
        default=None,
        description="Confidence score for the label (0.0-1.0). Set by vision refinement."
    )
```

### Vision provider interface

Add a `VisionLabelProvider` protocol that the scanner can call:

```python
class VisionLabelProvider(Protocol):
    async def refine_labels(
        self,
        page_image: bytes,       # PNG bytes of the rendered page
        fields: list[FormField],  # fields on this page with positions
        page_width: float,       # page width in PDF points
        page_height: float,      # page height in PDF points
    ) -> dict[str, str]:
        """Return a mapping of field_id -> label string."""
        ...
```

The default implementation will use the same LLM provider config that FormBridge already has (`formbridge[all]` extras, `llm.py`'s `LLMConfig`). The vision call requires a multimodal model (e.g., `gpt-4o`, `claude-sonnet-4-20250514`).

### Annotated image construction

For each page, construct an annotated PNG:

1. Render page at 200 DPI using `pdfplumber`'s page image export (`.to_image(resolution=200)`)
2. For each field on the page, draw a numbered rectangle at the field's position (converting PDF points to pixel coordinates)
3. Include a legend mapping numbers to field IDs

The prompt asks the vision model to return a JSON object mapping each field number to the printed text that labels that field on the form (e.g., `"42": "Wages, salaries, tips (W-2, box 1)"`).

### Opt-in and configuration

Vision refinement is opt-in. The scanner gains a `vision_labels` parameter:

```python
scanner = Scanner(pdf_path, vision_labels=True)
schema = scanner.scan()
```

CLI flag: `formbridge scan --vision-labels`

MCP tool parameter: `scan_form(pdf_path, vision_labels=False)`

When `vision_labels=False` (default), behavior is identical to today. This avoids adding an LLM dependency to the default scan path.

## Alternatives Considered

1. **Per-field vision calls** -- Render a crop around each field, call vision per field. Rejected: Form 1040 has 263 fields across 2 pages. Per-field calls would be 263 vision API calls vs. 2. The cost and latency are unacceptable.

2. **Per-field proximity text extraction (pdfplumber)** -- Extract text near each field's bounding box using pdfplumber's text extraction. Rejected: This is essentially what `_extract_label()` already attempts, and it produces garbled output on IRS forms because the PDFs use non-standard ToUnicode CMaps. The text layer is unreliable regardless of extraction tool.

3. **Batch page-level vision with annotated field positions (chosen)** -- Render each page once, annotate field bounding boxes, ask vision model to label all fields in a single call. 2 pages = 2 vision calls. Cost is manageable. The vision model sees the full form context, which helps it understand label structure (e.g., "this is a line number, not random text").

4. **External post-processing (let the MCP caller do vision)** -- Export scan results with positions but no labels, let the downstream tool (Hermes) render pages and do vision. Rejected: This pushes the complexity onto every consumer. FormBridge is the right place to own field identification. Also, the annotated-image + structured-prompt pattern is reusable across all FormBridge use cases, not just tax forms.

5. **Use the existing LLM-based mapper to infer labels** -- Run the mapper against a dummy dataset and see which fields it guesses. Rejected: This is circular. The mapper's accuracy depends on having good labels. Asking the mapper to produce labels from bad labels is unreliable.

### External Tooling Assessment (2026-04-06)

We evaluated several external document processing tools to determine if they could replace or supplement the vision-LLM approach:

**Chandra OCR 2 (Datalab)** — Open-source OCR model (Apache 2.0) that beats GPT-5 Mini and Gemini at document OCR (85.9% benchmark accuracy). Handles forms with checkboxes, tables, handwriting, 90 languages. Runs locally at ~2 pages/second on GPU, or via free hosted API. Install: `pip install chandra-ocr[all]`.

Assessment: Chandra is a strong text extraction tool but solves a different problem. It extracts *document content* but does not know about AcroForm field widgets or their bounding boxes. We would still need to correlate Chandra's text output with our field positions. Could replace pdfplumber's broken text extraction in the proximity-based label fallback, but does not eliminate the need for positional correlation. Not a fit for ADR 001's core vision approach, but could be useful as a future improvement to the non-vision label extraction path.

**LlamaParse / LiteParse (LlamaIndex)** — LlamaParse is a cloud API for multimodal PDF parsing (structured Markdown/JSON output). LiteParse (released March 2026) is the local, no-GPU alternative.

Assessment: Both extract document content (what is printed on the page) but have no awareness of AcroForm field widgets. They would produce clean text where pdfplumber produces garbage, but we would still need to correlate text positions with field bounding boxes. LlamaParse adds a cloud dependency and cost. LiteParse could theoretically replace pdfplumber as the text extraction backend for proximity matching, but the ADR 001 vision approach is simpler and more direct for our specific problem (mapping bounding boxes to semantic labels in 2 API calls).

**Docling (IBM)** — Open-source (MIT, 57K GitHub stars) PDF-to-Markdown/JSON converter with vision-based table recognition. Python-first.

Assessment: Docling has an open issue (#673) requesting AcroForm/form-field extraction support, which is not yet implemented. Like the others, it extracts content but not form field metadata. Would need positional correlation. Not a fit for our current needs.

**Unstructured.io** — Document ETL platform with VLM-based partitioning and semantic chunking. More oriented toward RAG ingestion pipelines than form field labeling.

Assessment: Overkill for our use case. The partitioning model produces semantic elements (titles, lists, tables) but not "which AcroForm field does this text label?" answers.

**Conclusion:** None of these tools solve the core problem (correlating AcroForm field bounding boxes with their semantic labels on rendered forms). The ADR 001 approach of rendering annotated pages and asking a vision LLM remains the most direct solution. However, Chandra OCR 2 or LiteParse could be valuable future additions to improve the non-vision fallback path (replacing pdfplumber's garbled text extraction for proximity matching).

## Consequences

- **Pros:**
  - Field labels become semantically meaningful on forms where pdfplumber produces garbage
  - Batch page-level rendering keeps the cost to N vision calls (one per page), not N-per-field
  - Opt-in preserves the existing zero-dependency scan path
  - Label provenance (`label_source`, `label_confidence`) enables downstream consumers to decide whether to trust the label
  - Improves mapper accuracy because the LLM gets human-readable labels instead of `field_042`

- **Cons:**
  - Vision refinement requires a multimodal LLM provider, adding a runtime dependency when enabled
  - The annotated image + structured prompt may not perfectly handle every form layout (nested subforms, multi-page tables)
  - Adds async to the scanner (vision calls are I/O-bound). The scanner is currently synchronous
  - PNG rendering and upload increases memory and bandwidth usage proportional to page count and resolution

## Technical Details

### Annotated image example

For Form 1040 page 1 with 150 fields, the annotated image would show the full form with small numbered rectangles overlaid on each fillable area. The prompt:

```
You are looking at a PDF form with numbered rectangles marking fillable fields.
For each numbered field, identify the printed label or description on the form.

Return a JSON object mapping field numbers to their labels.
Example: {"1": "First name and middle initial", "2": "Last name", "3": "Your social security number"}

If a field has no visible label (e.g., it's a calculation field or continuation area),
use the line number or a short description of what appears nearest to it.

Fields on this page:
- [1] field_001 (text) at position (33.8, 183.0)
- [2] field_002 (text) at position (33.8, 213.0)
...
```

### Coordinate conversion

PDF coordinates (points, origin bottom-left) to pixel coordinates (origin top-left):

```python
def pdf_to_pixel(pdf_x, pdf_y, pdf_width, pdf_height, img_width, img_height):
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height
    px = pdf_x * scale_x
    py = (pdf_height - pdf_y) * scale_y  # flip Y axis
    return px, py
```

### Integration point in scanner.py

```python
class Scanner:
    def __init__(self, pdf_path, verbose=False, vision_labels=False):
        self.vision_labels = vision_labels
        # ...

    def scan(self) -> FormSchema:
        # ... existing extraction ...
        schema = FormSchema(...)

        if self.vision_labels and schema.fields:
            self._refine_labels_with_vision(schema)

        return schema

    def _refine_labels_with_vision(self, schema: FormSchema) -> None:
        """Refine field labels using vision OCR on annotated page renders."""
        # Group fields by page
        # For each page: render, annotate, call vision, update labels
        ...
```

### Garbled label examples from current extraction

These are actual labels produced by the current scanner on IRS Form 1040:

```
field_022  label='Aif trteaqcuhi rSecdh.. B'
field_024  label='FFoo1r th0e ye4ar Ja0n.'
field_038  label='dsah'
field_039  label='tdsah'
field_040  label='Itdfh'
field_042  label='D(s'
field_046  label='one box. Manadr rfiuelldn nfialimnge sheepraer'
field_049  label='For FOtihltehede y rpeuarrs uJaannt. t1o– sDeect'
field_050  label='Filing StatusSMina'
field_052  label='nutrsy name S i n g le'
```

The corresponding fields should have labels like "Your first name and middle initial", "Your last name", "Your social security number", "Filing status: Single", etc.

## Supersedes / Dependencies

- depends on: the existing `llm.py` LLM provider infrastructure for multimodal API calls
- related: `nilsyai/formbridge` upstream issue for improved label extraction (TBD whether to contribute upstream)
- updated: 2026-04-06 — added external tooling assessment (Chandra OCR 2, LlamaParse/LiteParse, Docling, Unstructured.io); confirmed vision-LLM approach remains the best fit
