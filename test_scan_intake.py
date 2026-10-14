"""Tests for scanned-report reading.

The geometry rejoining is the part that matters, and it is a pure function
over block dictionaries, so all of it is tested without AWS. The Textract
call itself is tested against a stub client.
"""

from __future__ import annotations

import pytest

from scan_intake import (
    ScanError,
    read_many,
    read_report_file,
    read_scan,
    rows_from_blocks,
)


def line(text: str, top: float, left: float, height: float = 0.02) -> dict:
    return {
        "BlockType": "LINE",
        "Text": text,
        "Geometry": {"BoundingBox": {"Top": top, "Left": left, "Height": height, "Width": 0.2}},
    }


class StubTextract:
    """Stands in for the Textract client so the pipeline can be tested."""

    def __init__(self, blocks, *, fail: bool = False):
        self.blocks = blocks
        self.fail = fail
        self.calls = 0

    def detect_document_text(self, Document):   # noqa: N803 - boto3's casing
        self.calls += 1
        if self.fail:
            raise RuntimeError("AccessDeniedException")
        return {"Blocks": self.blocks}


# --------------------------------------------------- rejoining by geometry


def test_a_two_column_row_is_rejoined_in_reading_order():
    blocks = [
        line("6.8 mmol/L", top=0.300, left=0.55),
        line("Serum potassium", top=0.301, left=0.10),
    ]
    assert rows_from_blocks(blocks) == ["Serum potassium   6.8 mmol/L"]


def test_three_columns_rejoin_left_to_right():
    blocks = [
        line("HIGH", top=0.300, left=0.80),
        line("Serum potassium", top=0.300, left=0.10),
        line("6.8 mmol/L", top=0.302, left=0.55),
    ]
    assert rows_from_blocks(blocks) == ["Serum potassium   6.8 mmol/L   HIGH"]


def test_separate_rows_are_not_merged():
    blocks = [
        line("Serum potassium", top=0.300, left=0.10),
        line("6.8 mmol/L", top=0.300, left=0.55),
        line("Haemoglobin", top=0.340, left=0.10),
        line("11.2 g/dL", top=0.340, left=0.55),
    ]
    assert rows_from_blocks(blocks) == [
        "Serum potassium   6.8 mmol/L",
        "Haemoglobin   11.2 g/dL",
    ]


def test_a_slightly_skewed_scan_still_pairs():
    # A page fed in crooked: the right column drifts down a little.
    blocks = [
        line("Histopathology (biopsy)", top=0.500, left=0.10),
        line("awaiting review", top=0.508, left=0.55),
    ]
    assert rows_from_blocks(blocks) == [
        "Histopathology (biopsy)   awaiting review"
    ]


def test_rows_come_back_top_to_bottom():
    blocks = [
        line("third", top=0.700, left=0.10),
        line("first", top=0.100, left=0.10),
        line("second", top=0.400, left=0.10),
    ]
    assert rows_from_blocks(blocks) == ["first", "second", "third"]


def test_blocks_without_geometry_are_kept_not_dropped():
    blocks = [
        line("Serum potassium", top=0.300, left=0.10),
        {"BlockType": "LINE", "Text": "no geometry here"},
    ]
    rows = rows_from_blocks(blocks)
    assert "Serum potassium" in rows[0]
    assert "no geometry here" in rows


def test_non_line_blocks_are_ignored():
    blocks = [
        {"BlockType": "PAGE", "Text": "should not appear"},
        {"BlockType": "WORD", "Text": "Serum"},
        line("Serum potassium", top=0.300, left=0.10),
    ]
    assert rows_from_blocks(blocks) == ["Serum potassium"]


def test_empty_text_is_skipped():
    blocks = [line("   ", top=0.1, left=0.1), line("real", top=0.2, left=0.1)]
    assert rows_from_blocks(blocks) == ["real"]


def test_no_blocks_gives_no_rows():
    assert rows_from_blocks([]) == []


def test_zero_height_lines_do_not_collapse_into_one_row():
    # A degenerate page. Rows must still separate rather than merging.
    blocks = [
        line("first", top=0.10, left=0.10, height=0.0),
        line("second", top=0.50, left=0.10, height=0.0),
    ]
    assert rows_from_blocks(blocks) == ["first", "second"]


# ------------------------------------------------------- the Textract call


def test_read_scan_reports_how_many_lines_were_rejoined():
    stub = StubTextract([
        line("Serum potassium", top=0.30, left=0.10),
        line("6.8 mmol/L", top=0.30, left=0.55),
        line("Haemoglobin", top=0.34, left=0.10),
        line("11.2 g/dL", top=0.34, left=0.55),
    ])
    result = read_scan(b"fake-pdf-bytes", client=stub)

    assert stub.calls == 1
    assert result.source == "textract"
    assert result.line_count == 4
    assert result.row_count == 2
    assert "4 printed lines rejoined into 2 rows" in result.detail
    assert result.text == "Serum potassium   6.8 mmol/L\nHaemoglobin   11.2 g/dL"


def test_no_rejoining_leaves_the_detail_empty():
    stub = StubTextract([line("one per row", top=0.3, left=0.1)])
    assert read_scan(b"x", client=stub).detail == ""


def test_a_textract_failure_is_reported_not_swallowed():
    stub = StubTextract([], fail=True)
    with pytest.raises(ScanError, match="Textract could not read"):
        read_scan(b"x", client=stub)


def test_an_oversized_file_is_refused_before_it_is_sent():
    stub = StubTextract([])
    with pytest.raises(ScanError, match="limit is"):
        read_scan(b"x" * (6 * 1024 * 1024), client=stub)
    assert stub.calls == 0, "an oversized file must not be billed for"


# ------------------------------------------------------------ file routing


def test_a_text_file_is_read_without_ocr(tmp_path):
    path = tmp_path / "ward.txt"
    path.write_text("Serum potassium 6.8 mmol/L\nHaemoglobin 11.2 g/dL\n", encoding="utf-8")

    result = read_report_file(path)
    assert result.source == "text"
    assert result.line_count == 2
    assert "no OCR" in result.detail


def test_a_csv_is_treated_as_text(tmp_path):
    path = tmp_path / "ward.csv"
    path.write_text("test,status\nPotassium,released\n", encoding="utf-8")
    assert read_report_file(path).source == "text"


def test_a_pdf_goes_to_textract(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    stub = StubTextract([line("Serum potassium", top=0.3, left=0.1)])

    result = read_report_file(path, client=stub)
    assert result.source == "textract"
    assert stub.calls == 1


def test_an_unsupported_format_names_what_is_supported(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(b"not a report")
    with pytest.raises(ScanError, match="Supported:"):
        read_report_file(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(ScanError, match="No file at"):
        read_report_file(tmp_path / "nothing.txt")


def test_one_bad_file_does_not_stop_a_batch(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("Potassium released\n", encoding="utf-8")
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"x")

    results = read_many([good, bad, tmp_path / "gone.txt"])
    assert [row["ok"] for row in results] == [True, False, False]
    assert results[0]["file"] == "good.txt"
    assert "Supported:" in results[1]["error"]
    assert "No file at" in results[2]["error"]


# --------------------------------------------- the handover to intake.py


def test_rejoined_text_reaches_intake_with_values_stripped():
    """The whole point: OCR recovers a value, intake deletes it."""
    from pathlib import Path

    from carechaser_engine import load_workflow
    from intake import RuleExtractor, extract_report

    workflow = load_workflow(Path(__file__).parent / "workflow.yaml")

    stub = StubTextract([
        line("Serum potassium", top=0.30, left=0.10),
        line("6.8 mmol/L", top=0.30, left=0.55),
        line("released, awaiting review", top=0.30, left=0.75),
    ])
    scanned = read_scan(b"x", client=stub)
    assert "6.8" in scanned.text, "OCR should recover the value"

    result = extract_report(scanned.text, workflow, RuleExtractor())
    names = " ".join(row["test_name"] for row in result["rows"])

    assert "otassium" in names, "the test should survive"
    assert "6.8" not in names, "no clinical value may survive intake"
    assert "mmol" not in names
