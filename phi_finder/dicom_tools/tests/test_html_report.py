import pydicom
from pydicom.data import get_testdata_files

from phi_finder.dicom_tools import anonymise_dicom, html_report


def test_render_text_diff_marks_removed_and_inserted():
    # Removed PHI is struck through (<del>), the placeholder is <ins>, and the
    # unchanged surrounding text is preserved verbatim.
    html = html_report._render_text_diff(
        "Report for John Smith today.", "Report for XXXX today."
    )
    assert "<del>John Smith</del>" in html
    assert "<ins>XXXX</ins>" in html
    assert html.startswith("Report for ")
    assert html.endswith(" today.")


def test_render_text_diff_escapes_and_preserves_whitespace():
    # HTML-sensitive characters are escaped, and original whitespace (newlines)
    # survives so the note's layout is retained under white-space: pre-wrap.
    html = html_report._render_text_diff("a <b>&\nMr X", "a <b>&\nXXXX")
    assert "&lt;b&gt;&amp;" in html  # escaped, not raw markup
    assert "\n" in html  # newline preserved
    assert "<del>Mr X</del>" in html


def test_snapshot_and_collect_note_diffs():
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    note = (
        "CT BRAIN - CLINICAL DATA. Patient John Smith, 82 year old male. "
        "Headache since insertion six months ago."
    )
    ds.add_new(0x001021B0, "LT", note)  # Additional Patient History (long)
    ds.add_new(0x00080030, "TM", "120000")  # non-text VR, ignored

    snapshot = html_report.snapshot_long_text(ds)
    # Simulate the in-place redaction anonymise_image performs.
    ds[0x001021B0].value = (
        "CT BRAIN - CLINICAL DATA. Patient XXXX, XXXX. "
        "Headache since insertion six months ago."
    )
    diffs = html_report.collect_note_diffs(snapshot, ds)

    assert len(diffs) == 1
    assert diffs[0]["name"] == "Additional Patient History"
    assert diffs[0]["original"] == note
    assert "John Smith" not in diffs[0]["redacted"]


def test_collect_note_diffs_ignores_short_and_unchanged_fields():
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    ds.add_new(0x00081030, "LO", "Short desc")  # below the length threshold
    long_note = "X" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 5)
    ds.add_new(0x001021B0, "LT", long_note)  # long but will be left unchanged

    snapshot = html_report.snapshot_long_text(ds)
    ds[0x00081030].value = "XXXX"  # changed but too short to be a note
    diffs = html_report.collect_note_diffs(snapshot, ds)

    assert diffs == []


def test_collect_note_diffs_reaches_into_sequences():
    # Notes nested inside a sequence item are diffed too.
    ds = pydicom.Dataset()
    item = pydicom.Dataset()
    note = "A" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 3)
    item.add_new(0x001021B0, "LT", note)
    ds.add_new(0x00081115, "SQ", pydicom.Sequence([item]))  # Referenced Series Sequence

    snapshot = html_report.snapshot_long_text(ds)
    ds[0x00081115].value[0][0x001021B0].value = "XXXX"
    diffs = html_report.collect_note_diffs(snapshot, ds)

    assert len(diffs) == 1
    assert diffs[0]["original"] == note


def test_collect_note_diffs_labels_nested_location():
    # A nested note is reported with its full path, not just its element name.
    ds = pydicom.Dataset()
    item = pydicom.Dataset()
    note = "A" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 3)
    item.add_new(0x001021B0, "LT", note)
    ds.add_new(0x00081115, "SQ", pydicom.Sequence([item]))

    snapshot = html_report.snapshot_long_text(ds)
    ds[0x00081115].value[0][0x001021B0].value = "XXXX"
    diffs = html_report.collect_note_diffs(snapshot, ds)

    assert diffs[0]["location"] == (
        "Referenced Series Sequence [0] > Additional Patient History"
    )
    assert diffs[0]["removed"] is False


def test_collect_note_diffs_flags_removed_element():
    # A note that vanished with its enclosing sequence is flagged as removed,
    # not reported as an in-place redaction that happened to blank everything.
    ds = pydicom.Dataset()
    item = pydicom.Dataset()
    note = "B" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 3)
    item.add_new(0x001021B0, "LT", note)
    ds.add_new(0x00081115, "SQ", pydicom.Sequence([item]))

    snapshot = html_report.snapshot_long_text(ds)
    ds[0x00081115].value = pydicom.Sequence([])  # the PS3.15 "D" action
    diffs = html_report.collect_note_diffs(snapshot, ds)

    assert len(diffs) == 1
    assert diffs[0]["removed"] is True
    assert diffs[0]["redacted"] == ""


def test_collect_note_diffs_separates_duplicate_note_copies():
    # Regression: an SR carrying the same text at the root and inside Content
    # Sequence must report both copies separately -- the nested one removed
    # with the sequence, the root one redacted in place.
    ds = pydicom.Dataset()
    note = "CT BRAIN. Patient John Smith, 82 year old male, presented today."
    inner = pydicom.Dataset()
    inner.add_new(0x0040A160, "UT", note)
    outer = pydicom.Dataset()
    outer.add_new(0x0040A730, "SQ", pydicom.Sequence([inner]))
    ds.add_new(0x0040A160, "UT", note)
    ds.add_new(0x0040A730, "SQ", pydicom.Sequence([outer]))

    snapshot = html_report.snapshot_long_text(ds)
    ds[0x0040A730].value = pydicom.Sequence([])  # profile empties the sequence
    ds[0x0040A160].value = "CT BRAIN. Patient XXXX, XXXX, presented today."
    diffs = html_report.collect_note_diffs(snapshot, ds)

    by_location = {d["location"]: d for d in diffs}
    assert set(by_location) == {
        "Text Value",
        "Content Sequence [0] > Content Sequence [0] > Text Value",
    }
    assert by_location["Text Value"]["removed"] is False
    assert "John Smith" not in by_location["Text Value"]["redacted"]
    nested = by_location["Content Sequence [0] > Content Sequence [0] > Text Value"]
    assert nested["removed"] is True


def test_build_html_report_marks_removed_and_redacted_distinctly():
    diffs = [
        {
            "name": "Text Value",
            "location": "Text Value",
            "original": "Patient John Smith presented with a headache today.",
            "redacted": "Patient XXXX presented with a headache today.",
            "removed": False,
        },
        {
            "name": "Text Value",
            "location": "Content Sequence [4] > Text Value",
            "original": "Patient John Smith presented with a headache today.",
            "redacted": "",
            "removed": True,
        },
    ]
    html = html_report.build_html_report([], n_images=1, note_diffs=diffs)

    # Both copies are shown, told apart by location rather than by name alone.
    assert "Content Sequence [4] &gt; Text Value" in html
    assert "removed entirely" in html
    assert "redacted in place" in html
    # The removed one is not rendered as a word-level redaction.
    assert html.count("<ins>XXXX</ins>") == 1


def test_build_html_report_dedupes_per_location_not_per_name():
    # Same name, different place in the tree: both must survive de-duplication.
    base = {
        "name": "Text Value",
        "original": "Patient John Smith presented with a headache today.",
        "redacted": "",
        "removed": True,
    }
    diffs = [
        {**base, "location": "Content Sequence [1] > Text Value"},
        {**base, "location": "Content Sequence [4] > Text Value"},
        {**base, "location": "Content Sequence [4] > Text Value"},  # true duplicate
    ]
    html = html_report.build_html_report([], n_images=1, note_diffs=diffs)
    assert html.count("<h3>") == 2


def test_build_html_report_includes_note_diff():
    diffs = [
        {
            "name": "Additional Patient History",
            "original": "Patient John Smith presented with a headache today.",
            "redacted": "Patient XXXX presented with a headache today.",
        }
    ]
    html = html_report.build_html_report(
        [], n_images=3, session_id="S1", use_case="Standard", note_diffs=diffs
    )
    assert "Clinical notes" in html
    assert "Additional Patient History" in html
    assert "<del>John Smith</del>" in html
    assert "<ins>XXXX</ins>" in html


def test_build_html_report_dedupes_identical_note_diffs():
    diff = {
        "name": "Additional Patient History",
        "original": "Patient John Smith presented with a headache today.",
        "redacted": "Patient XXXX presented with a headache today.",
    }
    html = html_report.build_html_report([], n_images=2, note_diffs=[diff, dict(diff)])
    # The same note repeated across slices appears only once.
    assert html.count("Additional Patient History") == 1


def test_build_html_report_omits_diff_section_without_notes():
    html = html_report.build_html_report([], n_images=1, note_diffs=None)
    assert "Clinical notes" not in html
    assert "<del>" not in html
    assert 'class="diff"' not in html


def test_build_html_report_lists_and_dedupes_header_names():
    # Header names are de-duplicated (by tag) and escaped in the findings list.
    flagged = [
        {"tag": "(0010,0010)", "name": "Patient's Name"},
        {"tag": "(0010,0010)", "name": "Patient's Name"},  # duplicate slice
        {"tag": "(0010,0020)", "name": "Patient ID"},
    ]
    html = html_report.build_html_report(flagged, n_images=4)
    assert html.count("<li>") == 2
    assert "Patient&#x27;s Name" in html  # HTML-escaped
    assert "removed the following types" in html


def test_read_flagged_headers_round_trip():
    # anonymise_image records the flagged headers; read_flagged_headers reads them.
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    anonymised = anonymise_dicom.anonymise_image(ds, use_case="dicom_default")
    headers = html_report.read_flagged_headers(anonymised)
    assert isinstance(headers, list)
    assert all("name" in h and "tag" in h for h in headers)


def test_read_flagged_headers_absent_tag_returns_empty():
    # A dataset never touched by phi-finder yields an empty list, not an error.
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    assert html_report.read_flagged_headers(ds) == []
