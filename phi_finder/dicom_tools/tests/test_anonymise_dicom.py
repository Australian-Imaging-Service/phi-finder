import pytest
import pydicom
import json
from pydicom.data import get_testdata_files
from pydicom.valuerep import PersonName


from phi_finder.dicom_tools import anonymise_dicom


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
        anonymizer=anonymise_dicom.AnonymizerEngine(),
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
    anonymizer = anonymise_dicom.AnonymizerEngine()
    analyzer_results = analyser.analyze(
                    text=test_string, language="en", score_threshold=0.5
                )
    anonymized_text = anonymizer.anonymize(
        text=test_string,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": anonymise_dicom.OperatorConfig("replace", {"new_value": "[XXXX]"})
        },
    ).text
    assert "[XXXX]" in anonymized_text


TEST_STRINGS_CLEAN = ["Not sensitive", "Flat tire", "Most common"]
@pytest.mark.parametrize("test_string", TEST_STRINGS_CLEAN)
def test_presidio_regex_clean(test_string: str):
    analyser = anonymise_dicom._build_presidio_analyser(0.5)
    anonymizer = anonymise_dicom.AnonymizerEngine()
    analyzer_results = analyser.analyze(
                    text=test_string, language="en", score_threshold=0.5
                )
    anonymized_text = anonymizer.anonymize(
        text=test_string,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": anonymise_dicom.OperatorConfig("replace", {"new_value": "[XXXX]"})
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
    # The retain option is recorded in the method code sequence.
    assert anonymised_dataset.PatientIdentityRemoved == "YES"
    codes = [item.CodeValue for item in anonymised_dataset.DeidentificationMethodCodeSequence]
    assert "113100" in codes  # Basic Application Confidentiality Profile
    assert "113108" in codes  # Retain Patient Characteristics Option


def test_is_ps3_15_use_case_matching():
    from phi_finder.dicom_tools import ps3_15

    for value in ("PS3.15", "ps3.15", " PS3.15 ", "PS3_15", "ps3-15",
                  "PS3.15_Rtn. Pat.", "PS3.15 Retain Patient Characteristics",
                  "dicom_default", "DICOM Default", "dicom_retain_patient"):
        assert ps3_15.is_ps3_15_use_case(value)
    for value in (None, "", "Aggressive", "Standard", "PS3.14"):
        assert not ps3_15.is_ps3_15_use_case(value)


def test_retain_patient_characteristics_matching():
    from phi_finder.dicom_tools import ps3_15

    for value in ("PS3.15_Rtn. Pat.", "ps3.15 rtn pat",
                  "PS3.15 Retain Patient Characteristics",
                  "dicom_retain_patient", "dicom_retain_patient_characteristics"):
        assert ps3_15.retain_patient_characteristics(value)
    # Plain PS3.15 and non-PS3.15 use cases do not retain.
    for value in (None, "", "PS3.15", "ps3.15", "Standard", "dicom_default"):
        assert not ps3_15.retain_patient_characteristics(value)


def test_scan_private_headers_matching():
    from phi_finder.dicom_tools import ps3_15

    # The scan-private variants are still PS3.15 use cases, and the base variant
    # (plain vs retain) is still detected through the suffix.
    for value in ("dicom_default_scan_private",
                  "dicom_retain_patient_scan_private",
                  "PS3.15_scan_private"):
        assert ps3_15.scan_private_headers(value)
        assert ps3_15.is_ps3_15_use_case(value)
    assert ps3_15.retain_patient_characteristics("dicom_retain_patient_scan_private")
    assert not ps3_15.retain_patient_characteristics("dicom_default_scan_private")
    # Non scan-private PS3.15 use cases and non-PS3.15 use cases are not
    # scan-private (the suffix only counts on a PS3.15 base).
    for value in (None, "", "dicom_default", "PS3.15", "Standard",
                  "standard_scan_private"):
        assert not ps3_15.scan_private_headers(value)


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


def test_ps3_15_uid_replacement_is_deterministic():
    # The same original UID must always map to the same replacement so that
    # Study/Series UIDs stay consistent across the files of a series.
    from phi_finder.dicom_tools import ps3_15

    filename = get_testdata_files("CT_small.dcm")[0]
    ds1 = pydicom.dcmread(filename)
    ds2 = pydicom.dcmread(filename)
    ps3_15.apply_basic_profile(ds1)
    ps3_15.apply_basic_profile(ds2)
    assert ds1.StudyInstanceUID == ds2.StudyInstanceUID
    assert ds1.SeriesInstanceUID == ds2.SeriesInstanceUID
    assert ds1.SOPInstanceUID == ds2.SOPInstanceUID


def test_ps3_15_u_action_on_sequence_replaces_uids_in_items():
    # Source Image Sequence (0008,2112) has action X/Z/U*: the sequence is
    # kept and the UIDs inside its items are replaced, not the SQ element
    # itself (which would fail: SQ values must be Datasets, not UID strings).
    from phi_finder.dicom_tools import ps3_15

    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    item = pydicom.Dataset()
    item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    item.ReferencedSOPInstanceUID = "1.2.3.4.5"
    dataset.SourceImageSequence = pydicom.Sequence([item])

    ps3_15.apply_basic_profile(dataset)

    assert "SourceImageSequence" in dataset
    item_after = dataset.SourceImageSequence[0]
    assert item_after.ReferencedSOPInstanceUID != "1.2.3.4.5"  # U
    assert item_after.ReferencedSOPInstanceUID == ps3_15._replace_uid("1.2.3.4.5")


def test_ps3_15_removes_private_tags_and_recurses_into_sq():
    from phi_finder.dicom_tools import ps3_15

    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    block = dataset.private_block(0x000B, "acme corp", create=True)
    block.add_new(0x01, "LO", "John Doe")
    nested = pydicom.Dataset()
    nested.PatientName = PersonName("Dr Jane Doe")
    nested.PatientID = "12345"
    # Anatomic Region Sequence has no Table E.1-1 action, so the sequence
    # itself survives and the profile must be applied inside its items.
    dataset.AnatomicRegionSequence = pydicom.Sequence([nested])
    # Request Attributes Sequence has action X: removed altogether.
    dataset.RequestAttributesSequence = pydicom.Sequence([pydicom.Dataset()])

    ps3_15.apply_basic_profile(dataset)

    assert (0x000B, 0x0010) not in dataset
    assert (0x000B, 0x1001) not in dataset
    assert "RequestAttributesSequence" not in dataset
    nested_after = dataset.AnatomicRegionSequence[0]
    assert str(nested_after.PatientName) == ""  # Z
    assert nested_after.PatientID == "XXXX"  # Z/D resolved to dummy
