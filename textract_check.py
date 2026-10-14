"""Can this account use Amazon Textract on a scanned report?

Makes a synthetic scanned lab report (an image of text, saved as a PDF, with
no text layer — exactly what a ward scanner produces), sends it to Textract,
and prints what comes back.

If this works, CareChaser can accept scanned PDFs. If it returns
AccessDeniedException the fix is an IAM policy; anything else and we fall back
to local OCR.

    python textract_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCAN_PDF = HERE / "textract_test_scan.pdf"

# Deliberately messy, the way a real ward printout reads.
REPORT_LINES = [
    "MUZAFFAR KHAN SURGICAL HOSPITAL",
    "LABORATORY REPORT  -  WARD 2",
    "",
    "Patient ref: MK-4471        Date: 13/10/2026",
    "",
    "Serum potassium ......... 6.8 mmol/L     HIGH",
    "Haemoglobin ............. 11.2 g/dL",
    "Histopathology (biopsy) . report released, awaiting review",
    "Chest X-ray ............. pending radiologist",
    "",
    "Released by: Lab Desk       Not yet acknowledged",
]


def make_scan(path: Path) -> None:
    """A page of rendered text with no text layer — a scan, for our purposes."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1240, 1754          # A4 at 150 dpi
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)

    font = None
    for candidate in (
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 34)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
        print("   (no monospace font found; using the default, OCR may be poorer)")

    y = 120
    for line in REPORT_LINES:
        draw.text((110, y), line, fill="black", font=font)
        y += 52

    page.save(path, "PDF", resolution=150.0)


def main() -> None:
    import boto3
    from botocore.exceptions import ClientError

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

    print("Textract check")
    print(f"region: {region}\n")

    print("1. Making a synthetic scanned report (image only, no text layer)")
    try:
        make_scan(SCAN_PDF)
    except ImportError:
        print("   Pillow is missing. Run:  pip install pillow")
        sys.exit(1)
    print(f"   wrote {SCAN_PDF.name}, {SCAN_PDF.stat().st_size} bytes")

    print("\n2. Sending it to Amazon Textract")
    client = boto3.client("textract", region_name=region)
    try:
        response = client.detect_document_text(
            Document={"Bytes": SCAN_PDF.read_bytes()}
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "?")
        print(f"   FAILED: {code}")
        print(f"   {exc.response.get('Error', {}).get('Message', exc)}")
        print()
        if code in {"AccessDeniedException", "UnrecognizedClientException"}:
            print("   Fix: attach AmazonTextractFullAccess to the IAM user.")
        else:
            print("   Not a permissions problem. We fall back to local OCR.")
        sys.exit(2)

    lines = [
        block["Text"]
        for block in response.get("Blocks", [])
        if block.get("BlockType") == "LINE"
    ]

    print(f"   OK — {len(lines)} lines read back\n")
    print("3. WHAT TEXTRACT READ")
    for line in lines:
        print(f"   {line}")

    print("\n4. WHAT THIS MEANS FOR CARECHASER")
    hits = [
        line for line in lines
        if any(word in line.lower() for word in
               ("potassium", "haemoglobin", "histopathology", "x-ray"))
    ]
    print(f"   {len(hits)} of 4 test lines recovered.")
    print("   Those lines go straight into intake.py, which strips every value")
    print("   before a case is created. Textract reads; it never decides.")


if __name__ == "__main__":
    main()
