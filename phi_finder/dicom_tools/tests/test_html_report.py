import json

import pydicom
from presidio_analyzer import RecognizerResult
from pydicom.data import get_testdata_files

from phi_finder.dicom_tools import anonymise_dicom, html_report, ps3_15


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


def test_snapshot_and_collect_value_diffs():
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    note = (
        "CT BRAIN - CLINICAL DATA. Patient John Smith, 82 year old male. "
        "Headache since insertion six months ago."
    )
    ds.add_new(0x001021B0, "LT", note)  # Additional Patient History (long)

    snapshot = html_report.snapshot_values(ds)
    # Simulate the in-place redaction anonymise_image performs.
    ds[0x001021B0].value = (
        "CT BRAIN - CLINICAL DATA. Patient XXXX, XXXX. "
        "Headache since insertion six months ago."
    )
    diffs = html_report.collect_value_diffs(snapshot, ds)

    assert len(diffs) == 1
    assert diffs[0]["name"] == "Additional Patient History"
    assert diffs[0]["original"] == note
    assert "John Smith" not in diffs[0]["redacted"]
    assert diffs[0]["note"] is True  # long enough to be shown as a note diff


def test_collect_value_diffs_reports_short_and_non_text_values():
    # Every changed value is reported, whatever its VR or length -- only long
    # ones are flagged as notes, so the report can tabulate the rest.
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    ds.add_new(0x00081030, "LO", "Head CT for John Smith")  # short text
    ds.add_new(0x00080030, "TM", "120000")  # non-text VR

    snapshot = html_report.snapshot_values(ds)
    ds[0x00081030].value = "Head CT for XXXX"
    ds[0x00080030].value = "000000"
    diffs = {d["name"]: d for d in html_report.collect_value_diffs(snapshot, ds)}

    assert set(diffs) == {"Study Description", "Study Time"}
    assert diffs["Study Description"]["original"] == "Head CT for John Smith"
    assert diffs["Study Description"]["note"] is False
    assert diffs["Study Time"]["redacted"] == "000000"
    assert diffs["Study Time"]["note"] is False


def test_collect_value_diffs_ignores_unchanged_fields():
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    long_note = "X" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 5)
    ds.add_new(0x001021B0, "LT", long_note)  # long but will be left unchanged

    snapshot = html_report.snapshot_values(ds)
    diffs = html_report.collect_value_diffs(snapshot, ds)

    assert diffs == []


def test_collect_value_diffs_ignores_pixel_and_bulk_data():
    # Pixel data and long numeric arrays are not values a reader could read as
    # PHI, and dumping them would swamp the report, so they are never diffed.
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    ds.add_new(0x00283006, "US", list(range(html_report._MAX_MULTIVALUE_ITEMS + 1)))

    snapshot = html_report.snapshot_values(ds)
    ds.PixelData = b"\x00" * 128
    ds[0x00283006].value = [0] * (html_report._MAX_MULTIVALUE_ITEMS + 1)
    diffs = html_report.collect_value_diffs(snapshot, ds)

    assert diffs == []


def test_collect_value_diffs_distinguishes_emptied_from_removed():
    # The PS3.15 "Z" action blanks a value in place; "X" deletes the field.
    ds = pydicom.Dataset()
    ds.add_new(0x00081030, "LO", "Head CT")
    ds.add_new(0x00081080, "LO", "Cardiology")  # Admitting Diagnoses Description

    snapshot = html_report.snapshot_values(ds)
    ds[0x00081030].value = ""
    del ds[0x00081080]
    diffs = {d["name"]: d for d in html_report.collect_value_diffs(snapshot, ds)}

    assert diffs["Study Description"]["removed"] is False
    assert diffs["Study Description"]["redacted"] == ""
    assert diffs["Admitting Diagnoses Description"]["removed"] is True


def test_collect_value_diffs_stamps_tag_and_source():
    # Each diff carries the element's tag and what de-identified it, read back
    # from the audit element anonymise_image wrote.
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    snapshot = html_report.snapshot_values(ds)
    anon = anonymise_dicom.anonymise_image(
        ds, use_case="dicom_default", spacy_model_name="en_core_web_sm"
    )
    diffs = {d["name"]: d for d in html_report.collect_value_diffs(snapshot, anon)}

    assert diffs["Patient's Name"]["tag"] == "(0010,0010)"
    assert diffs["Patient's Name"]["source"] == ps3_15.SOURCE_PS3_15


def test_collect_value_diffs_credits_the_sequence_a_value_vanished_with():
    # The audit record is keyed by tag alone, so a nested value that is gone is
    # credited to the sequence that held it, not to a same-named field
    # elsewhere in the tree that a different engine handled.
    ds = pydicom.Dataset()
    inner = pydicom.Dataset()
    inner.add_new(0x0040A160, "UT", "Reported by Dr John Smith.")
    ds.add_new(0x0040A730, "SQ", pydicom.Sequence([inner]))  # Content Sequence
    ds.add_new(0x0040A160, "UT", "Reported by Dr John Smith.")

    snapshot = html_report.snapshot_values(ds)
    ds[0x0040A730].value = pydicom.Sequence([])  # the profile removed it
    ds[0x0040A160].value = "Reported by XXXX."  # the NER models redacted it
    block = ds.private_block(0x0209, "phi-finder", create=True)
    block.add_new(0x00, "UT", json.dumps([
        {"tag": "(0040, a730)", "name": "Content Sequence",
         "source": ps3_15.SOURCE_PS3_15},
        {"tag": "(0040, a160)", "name": "Text Value",
         "source": anonymise_dicom.SOURCE_NER},
    ]))
    diffs = {d["location"]: d for d in html_report.collect_value_diffs(snapshot, ds)}

    assert diffs["Text Value"]["source"] == anonymise_dicom.SOURCE_NER
    nested = diffs["Content Sequence [0] > Text Value"]
    assert nested["source"] == ps3_15.SOURCE_PS3_15


def test_collect_value_diffs_reaches_into_sequences():
    # Notes nested inside a sequence item are diffed too.
    ds = pydicom.Dataset()
    item = pydicom.Dataset()
    note = "A" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 3)
    item.add_new(0x001021B0, "LT", note)
    ds.add_new(0x00081115, "SQ", pydicom.Sequence([item]))  # Referenced Series Sequence

    snapshot = html_report.snapshot_values(ds)
    ds[0x00081115].value[0][0x001021B0].value = "XXXX"
    diffs = html_report.collect_value_diffs(snapshot, ds)

    assert len(diffs) == 1
    assert diffs[0]["original"] == note


def test_collect_value_diffs_labels_nested_location():
    # A nested note is reported with its full path, not just its element name.
    ds = pydicom.Dataset()
    item = pydicom.Dataset()
    note = "A" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 3)
    item.add_new(0x001021B0, "LT", note)
    ds.add_new(0x00081115, "SQ", pydicom.Sequence([item]))

    snapshot = html_report.snapshot_values(ds)
    ds[0x00081115].value[0][0x001021B0].value = "XXXX"
    diffs = html_report.collect_value_diffs(snapshot, ds)

    assert diffs[0]["location"] == (
        "Referenced Series Sequence [0] > Additional Patient History"
    )
    assert diffs[0]["removed"] is False


def test_collect_value_diffs_flags_removed_element():
    # A note that vanished with its enclosing sequence is flagged as removed,
    # not reported as an in-place redaction that happened to blank everything.
    ds = pydicom.Dataset()
    item = pydicom.Dataset()
    note = "B" * (html_report._CLINICAL_NOTE_MIN_LENGTH + 3)
    item.add_new(0x001021B0, "LT", note)
    ds.add_new(0x00081115, "SQ", pydicom.Sequence([item]))

    snapshot = html_report.snapshot_values(ds)
    ds[0x00081115].value = pydicom.Sequence([])  # the PS3.15 "D" action
    diffs = html_report.collect_value_diffs(snapshot, ds)

    assert len(diffs) == 1
    assert diffs[0]["removed"] is True
    assert diffs[0]["redacted"] == ""


def test_collect_value_diffs_separates_duplicate_note_copies():
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

    snapshot = html_report.snapshot_values(ds)
    ds[0x0040A730].value = pydicom.Sequence([])  # profile empties the sequence
    ds[0x0040A160].value = "CT BRAIN. Patient XXXX, XXXX, presented today."
    diffs = html_report.collect_value_diffs(snapshot, ds)

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
    html = html_report.build_html_report([], n_images=1, value_diffs=diffs)

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
    html = html_report.build_html_report([], n_images=1, value_diffs=diffs)
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
        [], n_images=3, session_id="S1", use_case="Standard", value_diffs=diffs
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
    html = html_report.build_html_report([], n_images=2, value_diffs=[diff, dict(diff)])
    # The same note repeated across slices appears only once.
    assert html.count("Additional Patient History") == 1


def test_build_html_report_omits_diff_section_without_notes():
    html = html_report.build_html_report([], n_images=1, value_diffs=None)
    assert "Clinical notes" not in html
    assert "Changed fields" not in html
    assert "<del>" not in html
    assert 'class="diff"' not in html


def test_build_html_report_tabulates_short_value_changes():
    # Short values get a before/after row rather than a word-level diff, so
    # every changed field is visible, not just the clinical notes.
    diffs = [
        {"name": "Patient's Name", "location": "Patient's Name",
         "tag": "(0010,0010)", "original": "Smith^John", "redacted": "XXXX",
         "removed": False, "note": False, "source": ps3_15.SOURCE_PS3_15},
        {"name": "Patient's Birth Date", "location": "Patient's Birth Date",
         "tag": "(0010,0030)", "original": "19430607", "redacted": "19430101",
         "removed": False, "note": False, "source": anonymise_dicom.SOURCE_NER},
    ]
    html = html_report.build_html_report([], n_images=1, value_diffs=diffs)

    assert "Changed fields" in html
    assert "<th>Original value</th>" in html
    assert '<td class="before">Smith^John</td><td class="after">XXXX</td>' in html
    assert '<td class="before">19430607</td><td class="after">19430101</td>' in html
    # The tag and the engine that de-identified each field are shown too.
    assert '<td class="tag">(0010,0010)</td>' in html
    assert '<span class="pill pill-ps315">PS3.15</span>' in html
    assert '<span class="pill pill-ner">NER model</span>' in html
    # Tabulated values are not also rendered as clinical notes.
    assert "Clinical notes" not in html


def test_build_html_report_table_marks_emptied_and_removed():
    diffs = [
        {"name": "Study Description", "location": "Study Description",
         "original": "Head CT", "redacted": "", "removed": False, "note": False},
        {"name": "Institution Name", "location": "Institution Name",
         "original": "St Elsewhere", "redacted": "", "removed": True,
         "note": False},
    ]
    html = html_report.build_html_report([], n_images=1, value_diffs=diffs)

    assert '<td class="after"><em>(emptied)</em></td>' in html
    assert '<td class="after"><em>(field removed)</em></td>' in html


def test_build_html_report_caps_values_per_field_and_says_so():
    # A field that differs in every slice (e.g. a UID) must not fill the
    # report, and what was left out has to be stated, not silently dropped.
    n = html_report._MAX_VALUES_PER_FIELD + 3
    diffs = [
        {"name": "SOP Instance UID", "location": "SOP Instance UID",
         "original": f"1.2.3.{i}", "redacted": f"9.9.9.{i}",
         "removed": False, "note": False}
        for i in range(n)
    ]
    html = html_report.build_html_report([], n_images=n, value_diffs=diffs)

    assert html.count('<td class="before">') == html_report._MAX_VALUES_PER_FIELD
    assert "and 3 further distinct value(s)" in html
    assert "1.2.3.0" in html and "1.2.3.7" not in html


def test_build_html_report_folds_private_fields_away():
    # A scanner writes hundreds of private fields; they go in a collapsed block
    # so the standard fields stay readable, but they are still all there.
    diffs = [
        {"name": "Patient ID", "location": "Patient ID", "original": "MRN1",
         "redacted": "XXXX", "removed": False, "note": False, "private": False},
        {"name": "[Angle of first view]", "location": "[Angle of first view]",
         "original": "-718.07", "redacted": "", "removed": True,
         "note": False, "private": True},
    ]
    html = html_report.build_html_report([], n_images=1, value_diffs=diffs)

    main, _, private = html.partition("<details>")
    assert "MRN1" in main and "MRN1" not in private
    assert "-718.07" in private and "-718.07" not in main
    assert "1 private (manufacturer-defined) field(s)" in private


def test_collect_value_diffs_marks_private_tags():
    ds = pydicom.Dataset()
    ds.add_new(0x00081030, "LO", "Head CT")
    block = ds.private_block(0x0009, "ACME 1.0", create=True)
    block.add_new(0x01, "LO", "scanner note")

    snapshot = html_report.snapshot_values(ds)
    ds[0x00081030].value = "XXXX"
    ds[0x00091001].value = "XXXX"
    diffs = {d["name"]: d for d in html_report.collect_value_diffs(snapshot, ds)}

    assert diffs["Study Description"]["private"] is False
    # The private creator itself is never reported, only the block's data.
    assert [d["private"] for n, d in diffs.items() if n != "Study Description"] == [True]


def test_build_html_report_escapes_table_values():
    diffs = [
        {"name": "Patient Comments", "location": "Patient Comments",
         "original": "<script>alert(1)</script>", "redacted": "XXXX",
         "removed": False, "note": False},
    ]
    html = html_report.build_html_report([], n_images=1, value_diffs=diffs)
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_build_html_report_falls_back_to_names_without_values():
    # With no diffs to show, the same table is rendered without the two columns
    # that would reproduce the PHI: the fields are named, their values are not.
    flagged = [
        {"tag": "(0010, 0010)", "name": "Patient's Name",
         "source": ps3_15.SOURCE_PS3_15},
        {"tag": "(0010, 0010)", "name": "Patient's Name",  # duplicate slice
         "source": ps3_15.SOURCE_PS3_15},
        {"tag": "(0040, a160)", "name": "Text Value",
         "source": anonymise_dicom.SOURCE_NER},
    ]
    html = html_report.build_html_report(flagged, n_images=4)

    assert html.count("<tr><td>") == 2  # de-duplicated by tag
    assert "Patient&#x27;s Name" in html  # HTML-escaped
    assert '<td class="tag">(0010,0010)</td>' in html  # tidied for display
    assert "<th>Original value</th>" not in html
    assert '<span class="pill pill-ps315">PS3.15</span>' in html
    assert '<span class="pill pill-ner">NER model</span>' in html


def test_build_html_report_handles_headers_without_source():
    # Files anonymised before provenance was recorded still render; their
    # Profile column is a dash rather than a wrong attribution.
    flagged = [{"tag": "(0010, 0010)", "name": "Patient's Name"}]
    html = html_report.build_html_report(flagged, n_images=1)

    assert html.count("<tr><td>") == 1
    assert '<td class="profile"><span class="muted">&mdash;</span></td>' in html
    assert '<span class="pill' not in html  # no engine is credited


def test_build_html_report_no_findings_message_unchanged():
    html = html_report.build_html_report([], n_images=1)
    assert "found no personal or health information" in html
    assert "<tr><td>" not in html


def test_flagged_headers_record_their_source(monkeypatch):
    # anonymise_image stamps each record so the report can group them: the
    # PS3.15 profile handles the standard headers, and the NER models read the
    # free-text ones the profile has no action for.
    # A stub analyser stands in for Presidio: this file is the model-free tier,
    # and a real one would load a full spaCy pipeline just to find one name.
    class _StubAnalyser:
        def analyze(self, text, **kwargs):
            start = text.find("John Smith")
            if start < 0:
                return []
            return [RecognizerResult("PERSON", start, start + len("John Smith"), 0.9)]

    monkeypatch.setattr(
        anonymise_dicom, "_build_presidio_analyser",
        lambda score_threshold, spacy_model_name: _StubAnalyser(),
    )

    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    ds.add_new(0x0040A160, "UT", "CT BRAIN. Reported by Dr John Smith today.")
    anonymised = anonymise_dicom.anonymise_image(ds, use_case="dicom_default", spacy_model_name="en_core_web_sm")

    headers = html_report.read_flagged_headers(anonymised)
    sources = {h["name"]: h["source"] for h in headers}
    assert sources["Patient's Name"] == ps3_15.SOURCE_PS3_15
    assert sources["Text Value"] == anonymise_dicom.SOURCE_NER


def test_read_flagged_headers_round_trip():
    # anonymise_image records the flagged headers; read_flagged_headers reads them.
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    anonymised = anonymise_dicom.anonymise_image(ds, use_case="dicom_default", spacy_model_name="en_core_web_sm")
    headers = html_report.read_flagged_headers(anonymised)
    assert isinstance(headers, list)
    assert all("name" in h and "tag" in h for h in headers)


def test_read_flagged_headers_absent_tag_returns_empty():
    # A dataset never touched by phi-finder yields an empty list, not an error.
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    assert html_report.read_flagged_headers(ds) == []
