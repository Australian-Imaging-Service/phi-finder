import pydicom
from pydicom.data import get_testdata_files
from pydicom.valuerep import PersonName

from phi_finder.dicom_tools import ps3_15


def test_is_ps3_15_use_case_matching():
    for value in ("PS3.15", "ps3.15", " PS3.15 ", "PS3_15", "ps3-15",
                  "PS3.15_Rtn. Pat.", "PS3.15 Retain Patient Characteristics",
                  "dicom_default", "DICOM Default", "dicom_retain_patient"):
        assert ps3_15.is_ps3_15_use_case(value)
    for value in (None, "", "Aggressive", "Standard", "PS3.14"):
        assert not ps3_15.is_ps3_15_use_case(value)


def test_retain_patient_characteristics_matching():
    for value in ("PS3.15_Rtn. Pat.", "ps3.15 rtn pat",
                  "PS3.15 Retain Patient Characteristics",
                  "dicom_retain_patient", "dicom_retain_patient_characteristics"):
        assert ps3_15.retain_patient_characteristics(value)
    # Plain PS3.15 and non-PS3.15 use cases do not retain.
    for value in (None, "", "PS3.15", "ps3.15", "Standard", "dicom_default"):
        assert not ps3_15.retain_patient_characteristics(value)


def test_scan_private_headers_matching():
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


def test_ps3_15_uid_replacement_is_deterministic():
    # The same original UID must always map to the same replacement so that
    # Study/Series UIDs stay consistent across the files of a series.
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
