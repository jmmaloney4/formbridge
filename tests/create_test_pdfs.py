"""Create a minimal test PDF for basic testing."""

from pathlib import Path


def create_minimal_pdf() -> bytes:
    """Create a minimal valid PDF file.

    This creates a very simple PDF with no form fields,
    useful for testing basic scanner functionality without
    downloading external fixtures.
    """
    # Minimal valid PDF with one page
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Form) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000361 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF
"""
    return pdf_content


def create_fillable_pdf() -> bytes:
    """Create a minimal PDF with AcroForm fields.

    This creates a PDF with a single text field for testing
    form field extraction.
    """
    # PDF with AcroForm
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [6 0 R] /DR << /Font << /Helv 5 0 R >> >> /DA (/Helv 0 Tf 0 g) /NeedAppearances true >> >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> /XObject << >> >> /Annots [6 0 R] >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Form) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>
endobj
6 0 obj
<< /Type /Annot /Subtype /Widget /FT /Tx /Rect [100 650 300 670] /P 3 0 R /T (field_001) /V () /DV () /Ff 0 >>
endobj
xref
0 7
0000000000 65535 f
0000000009 00000 n
0000000230 00000 n
0000000295 00000 n
0000000486 00000 n
0000000581 00000 n
0000000704 00000 n
trailer
<< /Size 7 /Root 1 0 R >>
startxref
855
%%EOF
"""
    return pdf_content


if __name__ == "__main__":
    # Create test fixtures
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    (fixtures_dir / "minimal.pdf").write_bytes(create_minimal_pdf())
    (fixtures_dir / "fillable.pdf").write_bytes(create_fillable_pdf())

    print("Created test fixtures:")
    print(f"  - {fixtures_dir / 'minimal.pdf'}")
    print(f"  - {fixtures_dir / 'fillable.pdf'}")
