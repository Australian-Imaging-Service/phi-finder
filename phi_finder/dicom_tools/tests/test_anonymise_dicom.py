import pytest
import pydicom
import json
from pydicom.data import get_testdata_files
from pydicom.valuerep import PersonName


from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from phi_finder.dicom_tools import anonymise_dicom, utils


def test_anonymise_with_transformer():
    model = anonymise_dicom._build_transformer()
    text = "John Doe was born on 01/01/1980 and has a social security number of 123-45-6789."
    anon = anonymise_dicom._anonymise_with_transformer(model, text=text, threshold=0.01)
    assert anon != text
    assert "XXXX" in anon


def test_destroy_pixels():
    filename = get_testdata_files("CT_small.dcm")[0]
    dataset = pydicom.dcmread(filename)
    assert "PixelData" in dataset
    assert dataset.pixel_array.shape != (8, 8)
    assert all(v != 0 for v in dataset.pixel_array.flatten())
    anonymised_dataset = anonymise_dicom.destroy_pixels(dataset)
    assert anonymised_dataset.pixel_array.shape == (8, 8)
    assert all(v == 0 for v in anonymised_dataset.pixel_array.flatten())


def test_destroy_pixels_compressed_source():
    # RLE-compressed source: destroying pixels must not require decoding the
    # originals, and the output must be readable as uncompressed data.
    filename = get_testdata_files("MR_small_RLE.dcm")[0]
    dataset = pydicom.dcmread(filename)
    anonymised_dataset = anonymise_dicom.destroy_pixels(dataset)
    assert anonymised_dataset.file_meta.TransferSyntaxUID == pydicom.uid.ExplicitVRLittleEndian
    assert anonymised_dataset.pixel_array.shape == (8, 8)
    assert not anonymised_dataset.pixel_array.any()


def test_destroy_pixels_multiframe_colour_source():
    filename = get_testdata_files("color3d_jpeg_baseline.dcm")[0]
    dataset = pydicom.dcmread(filename)
    assert int(dataset.NumberOfFrames) > 1
    assert dataset.SamplesPerPixel == 3
    anonymised_dataset = anonymise_dicom.destroy_pixels(dataset)
    assert anonymised_dataset.pixel_array.shape == (8, 8)
    assert not anonymised_dataset.pixel_array.any()
    assert anonymised_dataset.SamplesPerPixel == 1
    assert "NumberOfFrames" not in anonymised_dataset


class _RaisingAnalyser:
    def analyze(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_anonymise_ds_fails_closed():
    # If analysis errors out, the value must be blanked, not left as-is.
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    original_patient_id = dataset.PatientID
    assert original_patient_id != ""
    anonymised_headers = []
    anonymise_dicom._anonymise_ds(
        dataset,
        analyser=_RaisingAnalyser(),
        anonymizer=AnonymizerEngine(),
        score_threshold=0.5,
        anonymised_headers=anonymised_headers,
    )
    assert dataset.PatientID == ""
    pid_tag_str = str(pydicom.tag.Tag(0x0010, 0x0020))
    assert any(e["tag"] == pid_tag_str for e in anonymised_headers)


class _RaisingModel:
    def predict_entities(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_anonymise_with_transformer_fails_closed():
    text, labels = anonymise_dicom._anonymise_with_transformer(
        _RaisingModel(), "John Doe, 42 Wallaby Way", return_entities=True
    )
    assert text == "XXXX"
    assert labels == []


def test_age_string_replaced_with_valid_sentinel():
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    dataset.PatientAge = "076Y"
    anonymised_dataset = anonymise_dicom.anonymise_image(
        dataset,
        analyser=None,
        anonymizer=None,
        image_redactor=None,
        score_threshold=0.5,
        gliner_pii=None,
        use_case="Standard",
    )
    assert anonymised_dataset.PatientAge == "000Y"
    flagged = json.loads(anonymised_dataset[0x0209, 0x1000].value)
    age_tag_str = str(pydicom.tag.Tag(0x0010, 0x1010))
    assert any(e["tag"] == age_tag_str for e in flagged)


def test_specific_character_set_untouched():
    # The charset declaration (0008,0005) is a CS value the postcode/ORG
    # recognisers match ("ISO 2022 IR 100" contains "2022"); redacting it
    # breaks text decoding for every non-Latin dataset.
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    dataset.SpecificCharacterSet = "ISO 2022 IR 100"
    anonymised_dataset = anonymise_dicom.anonymise_image(
        dataset,
        use_case="Standard",
    )
    assert anonymised_dataset.SpecificCharacterSet == "ISO 2022 IR 100"


def test_private_creator_untouched_in_standard_mode():
    # Private creators are block bookkeeping, not PHI, and spaCy flags
    # "SIEMENS" as an organisation: redacting the creator would corrupt the
    # creator-to-data mapping of the whole block (e.g. Siemens CSA headers).
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    block = dataset.private_block(0x0011, "SIEMENS CSA HEADER", create=True)
    block.add_new(0x01, "LO", "John Doe")
    priv_tag = block.get_tag(0x01)
    creator_tag = pydicom.tag.Tag(priv_tag.group, priv_tag.element >> 8)

    anonymised = anonymise_dicom.anonymise_image(dataset, use_case="Standard")

    assert str(anonymised[creator_tag].value) == "SIEMENS CSA HEADER"
    # The block's data elements are still scanned and scrubbed.
    assert "John Doe" not in str(anonymised[priv_tag].value)
    assert "XXXX" in str(anonymised[priv_tag].value)


def test_build_engines_scan_private(monkeypatch):
    # The "..._scan_private" variants run the NER pipeline over private
    # headers, so the engines must be built once up front — otherwise
    # anonymise_image rebuilds the spaCy analyser for every file in the
    # series — with the caller's spaCy model, and GLiNER when requested.
    analyser_builds = []

    def fake_analyser_builder(score_threshold, spacy_model_name):
        analyser_builds.append((score_threshold, spacy_model_name))
        return "analyser"

    monkeypatch.setattr(
        anonymise_dicom, "_build_presidio_analyser", fake_analyser_builder
    )
    monkeypatch.setattr(anonymise_dicom, "_build_transformer", lambda: "gliner")

    analyser, anonymizer, image_redactor, gliner_pii = utils._build_engines(
        use_case="dicom_default_scan_private",
        score_threshold=0.5,
        spacy_model_name="en_core_web_sm",
        destroy_pixels=True,
        use_transformers=True,
    )
    assert analyser == "analyser"
    assert analyser_builds == [(0.5, "en_core_web_sm")]
    assert anonymizer is not None
    assert image_redactor is None
    assert gliner_pii == "gliner"


def test_build_engines_plain_ps3_15(monkeypatch):
    # The plain PS3.15 profile never runs the NER pipeline on headers, so with
    # destroyed pixels none of the (expensive) engines may be built.
    def _fail(*args, **kwargs):
        raise AssertionError("engine builder should not be called")

    monkeypatch.setattr(anonymise_dicom, "_build_presidio_analyser", _fail)
    monkeypatch.setattr(anonymise_dicom, "_build_transformer", _fail)

    engines = utils._build_engines(
        use_case="PS3.15",
        score_threshold=0.5,
        spacy_model_name="en_core_web_md",
        destroy_pixels=True,
        use_transformers=True,
    )
    assert engines == (None, None, None, None)


def test_structural_cs_values_untouched():
    # ImageType's magnitude component 'M' must survive the gender recognizer.
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    anonymised_dataset = anonymise_dicom.anonymise_image(
        dataset,
        analyser=None,
        anonymizer=None,
        image_redactor=None,
        score_threshold=0.5,
        gliner_pii=None,
        use_case="Standard",
    )
    assert list(anonymised_dataset.ImageType) == ["ORIGINAL", "PRIMARY", "M", "ND"]
    assert anonymised_dataset.Modality == "CT"


TEST_STRINGS_PII = ["John Doe",
                    "Jane Smith",
                    "Female",
                    "Male",
                    "01/01/1980",
                    "F", "M", "19430617", "076Y"]
@pytest.mark.parametrize("test_string", TEST_STRINGS_PII)
def test_presidio_regex_sensitive(test_string: str):
    analyser = anonymise_dicom._build_presidio_analyser(0.5)
    anonymizer = AnonymizerEngine()
    analyzer_results = analyser.analyze(
                    text=test_string, language="en", score_threshold=0.5
                )
    anonymized_text = anonymizer.anonymize(
        text=test_string,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": OperatorConfig("replace", {"new_value": "[XXXX]"})
        },
    ).text
    assert "[XXXX]" in anonymized_text


TEST_STRINGS_CLEAN = ["Not sensitive", "Flat tire", "Most common"]
@pytest.mark.parametrize("test_string", TEST_STRINGS_CLEAN)
def test_presidio_regex_clean(test_string: str):
    analyser = anonymise_dicom._build_presidio_analyser(0.5)
    anonymizer = AnonymizerEngine()
    analyzer_results = analyser.analyze(
                    text=test_string, language="en", score_threshold=0.5
                )
    anonymized_text = anonymizer.anonymize(
        text=test_string,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": OperatorConfig("replace", {"new_value": "[XXXX]"})
        },
    ).text
    assert test_string == anonymized_text


def test_anonymise_image():
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    #dataset = pydicom.dcmread("0.dcm")
    anonymised_dataset = anonymise_dicom.anonymise_image(dataset,
                                                         analyser=None,
                                                         anonymizer=None,
                                                         image_redactor=None,
                                                         score_threshold=0.5,
                                                         gliner_pii=None,
                                                         use_case="Standard")
    assert anonymised_dataset.PatientName == PersonName('XXXX')
    #assert anonymised_dataset[0x0010, 0x0040].value != 'XXXX'  # Sex unchanged
    if anonymised_dataset[0x0010, 0x0030].value != '':
        assert anonymised_dataset[0x0010, 0x0030].value[4:] == '0101'  # Month and day fixed.
    else:
        assert anonymised_dataset[0x0010, 0x0030].value == ''  # Birthdate empty if not kept
    #assert anonymised_dataset[0x0008, 0x0020].value == '20040119'  # Study Date unchanged
    #assert anonymised_dataset[0x0010, 0x1010].value == '000Y'  # Unchanged
    assert (0x02091000) in anonymised_dataset
    assert anonymised_dataset[0x0209, 0x1000].name == '[Flagged Headers PHI-Finder]'
    assert anonymised_dataset[0x0209, 0x1000].VR == 'UT'
    assert json.loads(anonymised_dataset[0x0209, 0x1000].value)


def test_anonymise_ds_recurses_into_sq():
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])

    nested = pydicom.Dataset()
    nested.PatientName = PersonName("Dr Jane Doe")
    dataset.RequestAttributesSequence = pydicom.Sequence([nested])

    anonymised_dataset = anonymise_dicom.anonymise_image(
        dataset,
        analyser=None,
        anonymizer=None,
        image_redactor=None,
        score_threshold=0.5,
        gliner_pii=None,
        use_case="Standard",
    )

    assert anonymised_dataset.PatientName == PersonName("XXXX")

    nested_after = anonymised_dataset.RequestAttributesSequence[0]
    assert nested_after.PatientName == PersonName("XXXX")

    flagged = json.loads(anonymised_dataset[0x0209, 0x1000].value)
    pn_tag_str = str(pydicom.tag.Tag(0x0010, 0x0010))
    pn_entries = [e for e in flagged if e.get("tag") == pn_tag_str]
    assert len(pn_entries) >= 2


def test_anonymise_image_ps3_15_use_case():
    # In the 'PS3.15' use case the headers must be handled by the PS3.15
    # basic profile alone: the NER engines are never invoked, so a raising
    # analyser must not be a problem.
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    dataset.PatientAge = "076Y"
    original_sop_uid = dataset.SOPInstanceUID
    original_study_uid = dataset.StudyInstanceUID
    anonymised_dataset = anonymise_dicom.anonymise_image(
        dataset,
        analyser=_RaisingAnalyser(),
        anonymizer=None,
        image_redactor=None,
        score_threshold=0.5,
        gliner_pii=None,
        use_case="PS3.15",
    )
    assert str(anonymised_dataset.PatientName) == ""  # Z
    assert anonymised_dataset.PatientBirthDate == ""  # Z
    assert "PatientAge" not in anonymised_dataset  # X
    assert anonymised_dataset.SOPInstanceUID != original_sop_uid  # U
    assert anonymised_dataset.StudyInstanceUID != original_study_uid  # U
    assert anonymised_dataset.file_meta.MediaStorageSOPInstanceUID == anonymised_dataset.SOPInstanceUID
    assert anonymised_dataset.PatientIdentityRemoved == "YES"
    assert anonymised_dataset.LongitudinalTemporalInformationModified == "REMOVED"
    assert "PS3.15" in anonymised_dataset.DeidentificationMethod
    # The flagged-headers private block is still written.
    flagged = json.loads(anonymised_dataset[0x0209, 0x1000].value)
    pn_tag_str = str(pydicom.tag.Tag(0x0010, 0x0010))
    assert any(e["tag"] == pn_tag_str for e in flagged)


def test_anonymise_image_ps3_15_retain_patient_characteristics():
    # The Retain Patient Characteristics variant keeps patient characteristics
    # (age, sex, size, weight) while still removing direct identifiers.
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    dataset.PatientAge = "076Y"
    dataset.PatientSex = "F"
    dataset.PatientWeight = "70"
    dataset.PatientSize = "1.8"
    dataset.PatientBirthDate = "19500101"
    anonymised_dataset = anonymise_dicom.anonymise_image(
        dataset,
        analyser=_RaisingAnalyser(),
        anonymizer=None,
        image_redactor=None,
        score_threshold=0.5,
        gliner_pii=None,
        use_case="PS3.15_Rtn. Pat.",
    )
    # Patient characteristics retained.
    assert anonymised_dataset.PatientAge == "076Y"
    assert anonymised_dataset.PatientSex == "F"
    assert str(anonymised_dataset.PatientWeight) == "70"
    assert str(anonymised_dataset.PatientSize) == "1.8"
    # Direct identifiers still removed/emptied.
    assert str(anonymised_dataset.PatientName) == ""  # Z
    assert anonymised_dataset.PatientBirthDate == ""  # Z
    # The retain option is recorded in the method code sequence; dates are
    # still removed, so (0028,0303) is REMOVED in this variant too.
    assert anonymised_dataset.PatientIdentityRemoved == "YES"
    assert anonymised_dataset.LongitudinalTemporalInformationModified == "REMOVED"
    codes = [item.CodeValue for item in anonymised_dataset.DeidentificationMethodCodeSequence]
    assert "113100" in codes  # Basic Application Confidentiality Profile
    assert "113108" in codes  # Retain Patient Characteristics Option


def test_anonymise_image_scan_private_keeps_and_scrubs_private():
    # dicom_default_scan_private: standard headers follow the Basic Profile,
    # but private attributes are kept and NER-scrubbed instead of removed.
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    # A name is used as the private creator too, so the test exercises the guard
    # that keeps private creators intact (scrubbing one would corrupt the block).
    block = dataset.private_block(0x0011, "John Doe", create=True)
    block.add_new(0x01, "LO", "Jane Smith")
    priv_tag = block.get_tag(0x01)
    creator_tag = pydicom.tag.Tag(priv_tag.group, priv_tag.element >> 8)

    anonymised = anonymise_dicom.anonymise_image(
        dataset, use_case="dicom_default_scan_private"
    )

    # Standard headers are still de-identified by the Basic Profile.
    assert str(anonymised.PatientName) == ""  # Z
    assert anonymised.PatientIdentityRemoved == "YES"
    # The private data element is kept (not removed) but its PHI is scrubbed.
    assert priv_tag in anonymised
    assert "Jane Smith" not in str(anonymised[priv_tag].value)
    assert "XXXX" in str(anonymised[priv_tag].value)
    # The private creator is left intact so the block mapping survives.
    assert str(anonymised[creator_tag].value) == "John Doe"

    # Contrast: the plain profile removes private attributes entirely.
    plain = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    pblock = plain.private_block(0x0011, "John Doe", create=True)
    pblock.add_new(0x01, "LO", "Jane Smith")
    plain_anon = anonymise_dicom.anonymise_image(plain, use_case="dicom_default")
    assert pblock.get_tag(0x01) not in plain_anon


def test_render_text_diff_marks_removed_and_inserted():
    # Removed PHI is struck through (<del>), the placeholder is <ins>, and the
    # unchanged surrounding text is preserved verbatim.
    html = utils._render_text_diff(
        "Report for John Smith today.", "Report for XXXX today."
    )
    assert "<del>John Smith</del>" in html
    assert "<ins>XXXX</ins>" in html
    assert html.startswith("Report for ")
    assert html.endswith(" today.")


def test_render_text_diff_escapes_and_preserves_whitespace():
    # HTML-sensitive characters are escaped, and original whitespace (newlines)
    # survives so the note's layout is retained under white-space: pre-wrap.
    html = utils._render_text_diff("a <b>&\nMr X", "a <b>&\nXXXX")
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

    snapshot = utils._snapshot_long_text(ds)
    # Simulate the in-place redaction anonymise_image performs.
    ds[0x001021B0].value = (
        "CT BRAIN - CLINICAL DATA. Patient XXXX, XXXX. "
        "Headache since insertion six months ago."
    )
    diffs = utils._collect_note_diffs(snapshot, ds)

    assert len(diffs) == 1
    assert diffs[0]["name"] == "Additional Patient History"
    assert diffs[0]["original"] == note
    assert "John Smith" not in diffs[0]["redacted"]


def test_collect_note_diffs_ignores_short_and_unchanged_fields():
    ds = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    ds.add_new(0x00081030, "LO", "Short desc")  # below the length threshold
    long_note = "X" * (utils._CLINICAL_NOTE_MIN_LENGTH + 5)
    ds.add_new(0x001021B0, "LT", long_note)  # long but will be left unchanged

    snapshot = utils._snapshot_long_text(ds)
    ds[0x00081030].value = "XXXX"  # changed but too short to be a note
    diffs = utils._collect_note_diffs(snapshot, ds)

    assert diffs == []


def test_build_html_report_includes_note_diff():
    diffs = [
        {
            "name": "Additional Patient History",
            "original": "Patient John Smith presented with a headache today.",
            "redacted": "Patient XXXX presented with a headache today.",
        }
    ]
    html = utils.build_html_report(
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
    html = utils.build_html_report([], n_images=2, note_diffs=[diff, dict(diff)])
    # The same note repeated across slices appears only once.
    assert html.count("Additional Patient History") == 1


def test_build_html_report_omits_diff_section_without_notes():
    html = utils.build_html_report([], n_images=1, note_diffs=None)
    assert "Clinical notes" not in html
    assert "<del>" not in html
    assert 'class="diff"' not in html
