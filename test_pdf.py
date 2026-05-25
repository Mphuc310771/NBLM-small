"""
Diagnostic: test pdfplumber extraction against EVERY PDF in the uploads folder.
Prints raw tracebacks, page-by-page extraction details, and chunk counts.
"""
import io
import os
import sys
import traceback
import glob

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pdfplumber
from app.application.upload_usecase import UploadUseCase


class MockVectorStore:
    def __init__(self):
        self.stored_texts = []
    def add_documents(self, texts, metadatas=None):
        self.stored_texts.extend(texts)


def diagnose_pdf(filepath):
    print(f"\n{'='*70}")
    print(f"  FILE: {filepath}")
    print(f"  SIZE: {os.path.getsize(filepath):,} bytes")
    print(f"{'='*70}")

    with open(filepath, "rb") as f:
        raw_bytes = f.read()

    # --- Step 1: Raw pdfplumber extraction (no UploadUseCase) ---
    print("\n[PHASE 1] Raw pdfplumber extraction:")
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            print(f"  Total pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                tables = page.extract_tables()
                text_len = len(text) if text else 0
                num_tables = len(tables) if tables else 0
                print(f"  Page {i+1}: text={text_len} chars, tables={num_tables}")

                # Inspect table cells for None/non-string values
                if tables:
                    for ti, table in enumerate(tables):
                        print(f"    Table {ti+1}: {len(table)} rows")
                        for ri, row in enumerate(table[:3]):  # first 3 rows only
                            cell_types = [f"{type(c).__name__}:{repr(c)[:30]}" for c in (row or [])]
                            print(f"      Row {ri}: [{', '.join(cell_types)}]")
                        if len(table) > 3:
                            print(f"      ... ({len(table)-3} more rows)")
    except Exception:
        print("  EXCEPTION during raw extraction:")
        traceback.print_exc()

    # --- Step 2: Full UploadUseCase pipeline ---
    print("\n[PHASE 2] UploadUseCase pipeline:")
    try:
        mock = MockVectorStore()
        uc = UploadUseCase(vector_store=mock)
        result = uc.execute(raw_bytes, os.path.basename(filepath), notebook_id="diag")
        print(f"  Result: {result}")
        print(f"  Chunks stored: {len(mock.stored_texts)}")
        if mock.stored_texts:
            print(f"  First chunk preview (200 chars):")
            print(f"    {mock.stored_texts[0][:200]}")
        else:
            print("  WARNING: 0 chunks! Extracting text directly for debug...")
            text = uc._extract_text(raw_bytes, os.path.basename(filepath))
            print(f"  Extracted text length: {len(text)}")
            if text:
                print(f"  Text preview (300 chars): {text[:300]}")
            else:
                print("  TEXT IS COMPLETELY EMPTY - pdfplumber returned nothing.")
    except Exception:
        print("  EXCEPTION during UploadUseCase:")
        traceback.print_exc()


if __name__ == "__main__":
    # Find all PDFs: first check uploads/, then check if any PDF path was passed as arg
    pdf_files = []

    # Check common upload dirs
    for search_dir in ["uploads", "app/uploads", "data", "."]:
        full_dir = os.path.join(os.path.dirname(__file__), search_dir)
        if os.path.isdir(full_dir):
            found = glob.glob(os.path.join(full_dir, "*.pdf"))
            pdf_files.extend(found)

    # Also accept CLI arg
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.isfile(arg) and arg.lower().endswith(".pdf"):
                pdf_files.append(arg)

    pdf_files = list(set(pdf_files))  # deduplicate

    if not pdf_files:
        print("No PDF files found. Pass a PDF path as argument:")
        print("  python test_pdf.py path/to/your_file.pdf")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) to diagnose.\n")
    for fp in sorted(pdf_files):
        diagnose_pdf(fp)

    print(f"\n{'='*70}")
    print("  DIAGNOSIS COMPLETE")
    print(f"{'='*70}")
