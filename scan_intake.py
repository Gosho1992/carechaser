"""Scanned reports into text CareChaser can read.

A ward printout is a scan: an image with no text layer. Amazon Textract reads
it. But Textract returns every *visual* line as its own block, so a two-column
report comes back with the test name and its value separated:

    Serum potassium
    6.8 mmol/L
    HIGH

intake.py works a line at a time, so that shape loses the pairing: the test
name arrives with no status, and the value arrives with no test. The fix is
geometry — Textract gives a bounding box per line, and lines printed at the
same height on the page belong to the same row:

    Serum potassium   6.8 mmol/L   HIGH

That rejoining is `rows_from_blocks`, which is a pure function over block
dictionaries and is tested without touching AWS.

What this module does NOT do: decide anything. Textract reads pixels into
characters. The rejoined text goes to intake.py, which strips every clinical
value out before a case is created. OCR never sees a workflow decision, and
the values it recovers are deleted on the way past.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

TEXT_SUFFIXES = {".txt", ".csv", ".md", ".log"}
SCAN_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

# Textract charges per page and the sync API takes one page at a time, so a
# file larger than this is refused rather than silently truncated.
MAX_SCAN_BYTES = 5 * 1024 * 1024


class ScanError(RuntimeError):
    """Raised when a file cannot be read. Never raised for an empty report."""


@dataclass(frozen=True)
class ScanResult:
    text: str
    source: str          # "text" | "textract"
    line_count: int
    row_count: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "line_count": self.line_count,
            "row_count": self.row_count,
            "detail": self.detail,
        }


# ------------------------------------------------------------- the geometry


def _box(block: dict[str, Any]) -> Optional[dict[str, float]]:
    geometry = block.get("Geometry")
    if not isinstance(geometry, dict):
        return None
    box = geometry.get("BoundingBox")
    return box if isinstance(box, dict) else None


def _centre(box: dict[str, float]) -> float:
    return float(box.get("Top", 0.0)) + float(box.get("Height", 0.0)) / 2


def rows_from_blocks(
    blocks: Sequence[dict[str, Any]], *, tolerance: float = 0.6
) -> list[str]:
    """Rejoin Textract LINE blocks printed at the same height into one row.

    `tolerance` is a fraction of the median line height: two lines whose
    vertical centres sit closer than that are the same row. 0.6 is forgiving
    enough for a slightly skewed scan and tight enough not to merge adjacent
    rows of a dense table.

    Blocks without geometry keep their own line, in the order given, rather
    than being dropped.
    """
    placed: list[tuple[float, float, str]] = []     # centre, left, text
    loose: list[str] = []

    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue
        text = str(block.get("Text", "")).strip()
        if not text:
            continue
        box = _box(block)
        if box is None:
            loose.append(text)
            continue
        placed.append((_centre(box), float(box.get("Left", 0.0)), text))

    if not placed:
        return loose

    heights = sorted(
        float(_box(b).get("Height", 0.0))                       # type: ignore[union-attr]
        for b in blocks
        if b.get("BlockType") == "LINE" and _box(b) is not None
    )
    median_height = heights[len(heights) // 2] if heights else 0.0
    # A degenerate page (every height zero) must not collapse into one row.
    gap = (median_height or 0.01) * tolerance

    placed.sort(key=lambda item: (item[0], item[1]))

    rows: list[list[tuple[float, str]]] = []
    anchor = placed[0][0]
    current: list[tuple[float, str]] = []

    for centre, left, text in placed:
        if current and abs(centre - anchor) > gap:
            rows.append(current)
            current = []
            anchor = centre
        elif not current:
            anchor = centre
        current.append((left, text))
    if current:
        rows.append(current)

    joined = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        joined.append("   ".join(text for _, text in row))
    return joined + loose


# ---------------------------------------------------------------- the reader


def _textract_client(region: Optional[str] = None):
    import boto3

    region = (
        region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    return boto3.client("textract", region_name=region)


def read_scan(data: bytes, *, client: Any | None = None) -> ScanResult:
    """One page of scan into rejoined text."""
    if len(data) > MAX_SCAN_BYTES:
        raise ScanError(
            f"That file is {len(data) // 1024} KB. The limit is "
            f"{MAX_SCAN_BYTES // 1024} KB for a single page."
        )

    client = client or _textract_client()
    try:
        response = client.detect_document_text(Document={"Bytes": data})
    except Exception as exc:                      # botocore raises many shapes
        raise ScanError(f"Textract could not read that file: {exc}") from exc

    blocks = response.get("Blocks", [])
    lines = [b for b in blocks if b.get("BlockType") == "LINE"]
    rows = rows_from_blocks(blocks)

    detail = ""
    if len(lines) > len(rows):
        detail = (
            f"{len(lines)} printed lines rejoined into {len(rows)} rows "
            f"by position on the page"
        )

    return ScanResult(
        text="\n".join(rows),
        source="textract",
        line_count=len(lines),
        row_count=len(rows),
        detail=detail,
    )


def read_report_file(
    path: str | Path, *, client: Any | None = None
) -> ScanResult:
    """Any supported report file into text intake.py can consume."""
    path = Path(path)
    if not path.exists():
        raise ScanError(f"No file at {path}")

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        return ScanResult(
            text=text,
            source="text",
            line_count=len(lines),
            row_count=len(lines),
            detail="read directly, no OCR needed",
        )

    if suffix in SCAN_SUFFIXES:
        return read_scan(path.read_bytes(), client=client)

    raise ScanError(
        f"{suffix or 'that file'} is not a report format CareChaser reads. "
        f"Supported: {', '.join(sorted(TEXT_SUFFIXES | SCAN_SUFFIXES))}"
    )


def read_many(
    paths: Iterable[str | Path], *, client: Any | None = None
) -> list[dict[str, Any]]:
    """A batch. One bad file reports itself and does not stop the rest."""
    out: list[dict[str, Any]] = []
    for path in paths:
        name = Path(path).name
        try:
            result = read_report_file(path, client=client)
            out.append({"file": name, "ok": True, **result.to_dict()})
        except ScanError as exc:
            out.append({"file": name, "ok": False, "error": str(exc)})
    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Read a report file")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = read_many(args.paths)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for row in results:
            print(f"\n=== {row['file']} ===")
            if not row["ok"]:
                print(f"  FAILED: {row['error']}")
                continue
            print(f"  source: {row['source']}  {row.get('detail', '')}")
            for line in row["text"].splitlines():
                print(f"  {line}")
