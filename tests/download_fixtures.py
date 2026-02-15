"""Test fixtures download script.

This script downloads IRS forms for testing. Run it once to set up test fixtures:

    python tests/download_fixtures.py

"""

import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# External forms to download (optional - for integration tests)
EXTERNAL_FORMS = [
    ("f1065.pdf", "https://www.irs.gov/pub/irs-pdf/f1065.pdf"),
    ("f1120.pdf", "https://www.irs.gov/pub/irs-pdf/f1120.pdf"),
    ("fw9.pdf", "https://www.irs.gov/pub/irs-pdf/fw9.pdf"),
]

# Minimal test PDFs (created locally - always available)
MINIMAL_PDFS = {
    "minimal.pdf": b"""%PDF-1.4
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
""",
    "fillable.pdf": b"""%PDF-1.4
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
""",
}


def create_minimal_fixtures() -> None:
    """Create minimal test PDFs that don't require downloading."""
    for filename, content in MINIMAL_PDFS.items():
        dest = FIXTURES_DIR / filename
        dest.write_bytes(content)
        print(f"✓ Created {filename}")


def download_fixtures() -> None:
    """Download test fixture PDFs from IRS."""
    FIXTURES_DIR.mkdir(exist_ok=True)

    # First create local test PDFs
    print("Creating minimal test PDFs...")
    create_minimal_fixtures()

    # Then try to download external forms (optional)
    print("\nDownloading IRS forms (optional)...")
    for filename, url in EXTERNAL_FORMS:
        dest = FIXTURES_DIR / filename
        if dest.exists():
            print(f"✓ {filename} already exists")
            continue

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print("  (This is optional - basic tests will still work)")


if __name__ == "__main__":
    download_fixtures()
