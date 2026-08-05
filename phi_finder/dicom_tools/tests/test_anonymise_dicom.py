import pytest
import pydicom
import json
import warnings
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


DESTROY_PIXELS_SOURCES = [
    "MR_small_implicit.dcm",
    "MR_small_bigendian.dcm",
    "MR_small_RLE.dcm",
    "CT_small.dcm",
]
@pytest.mark.parametrize("source", DESTROY_PIXELS_SOURCES)
def test_destroy_pixels_round_trips_through_save_as(source, tmp_path):
    dataset = pydicom.dcmread(get_testdata_files(source)[0])
    anonymised_dataset = anonymise_dicom.destroy_pixels(dataset)
    assert anonymised_dataset.is_implicit_VR is False
    assert anonymised_dataset.is_little_endian is True

    path = tmp_path / "anonymised.dcm"
    anonymised_dataset.save_as(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reloaded = pydicom.dcmread(path)
        pixels = reloaded.pixel_array

    mismatch = [str(w.message) for w in caught
                if "VR" in str(w.message) or "endian" in str(w.message)]
    assert not mismatch, mismatch
    assert reloaded.file_meta.TransferSyntaxUID == pydicom.uid.ExplicitVRLittleEndian
    assert pixels.shape == (8, 8)
    assert not pixels.any()


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


def test_build_engines_plain_ps3_15(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("engine builder should not be called")

    sentinel = object()
    monkeypatch.setattr(anonymise_dicom, "_build_presidio_analyser", _fail)
    monkeypatch.setattr(anonymise_dicom, "_build_transformer", lambda: sentinel)

    engines = utils._build_engines(
        use_case="PS3.15",
        score_threshold=0.5,
        spacy_model_name="en_core_web_md",
        destroy_pixels=True,
        use_transformers=False,
    )
    assert engines == (None, None, None, None)

    # GLiNER is built whenever it was asked for, so the free-text scan gets it.
    analyser, anonymizer, image_redactor, gliner_pii = utils._build_engines(
        use_case="PS3.15",
        score_threshold=0.5,
        spacy_model_name="en_core_web_md",
        destroy_pixels=True,
        use_transformers=True,
    )
    assert (analyser, anonymizer, image_redactor) == (None, None, None)
    assert gliner_pii is sentinel


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


SR_REPORT_TEXT = (
    " CT BRAIN - CLINICAL DATA Right facial droop, reported by Dr Emily Watson "
    "of Royal Melbourne Hospital on 04/03/2019. Patient John Smith, phone "
    "0412 345 678. TECHNIQUE Non-contrast axial images were acquired."
)


def _sr_dataset(text: str = SR_REPORT_TEXT) -> pydicom.dataset.Dataset:
    """Builds a minimal SR that carries its report in Text Value (0040,A160)."""
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    dataset.Modality = "SR"
    dataset.ValueType = "CONTAINER"
    dataset.TextValue = text
    return dataset


@pytest.mark.parametrize(
    "use_case", ["dicom_default", "dicom_retain_patient", "dicom_retain_patient_scan_private"]
)
def test_ps3_15_scans_sr_text_value(use_case):
    # Text Value (0040,A160) has no PS3.15 Table E.1-1 action, so the Basic
    # Profile leaves it alone. It holds the whole narrative report of an SR, so
    # every PS3.15 variant must run the NER pipeline over it rather than let it
    # through untouched.
    anonymised = anonymise_dicom.anonymise_image(_sr_dataset(), use_case=use_case)

    text_value = str(anonymised[0x0040, 0xA160].value)
    # The report itself is kept -- it is scrubbed, not removed.
    assert 0x0040A160 in anonymised
    assert "Non-contrast axial images were acquired." in text_value
    # ...but its PHI is gone.
    assert "John Smith" not in text_value
    assert "Emily Watson" not in text_value
    assert "Royal Melbourne Hospital" not in text_value
    assert "0412 345 678" not in text_value
    assert "04/03/2019" not in text_value
    assert "XXXX" in text_value
    # The change is recorded in the audit block like any other redacted header.
    flagged = json.loads(anonymised[0x0209, 0x1000].value)
    assert any(header["tag"] == "(0040, a160)" for header in flagged)


def test_ps3_15_without_free_text_builds_no_analyser(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("analyser should not be built")

    monkeypatch.setattr(anonymise_dicom, "_build_presidio_analyser", _fail)

    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    assert "TextValue" not in dataset
    anonymised = anonymise_dicom.anonymise_image(dataset, use_case="dicom_default")
    assert anonymised.PatientIdentityRemoved == "YES"
