import numpy as np
from pathlib import Path
from frametree.core.row import DataRow

from phi_finder.dicom_tools import utils


def test_data_row_access(tmp_path: Path, data_row: DataRow) -> None:
    assert len(data_row.entries) == 3
    assert isinstance(data_row, DataRow)


def test_secondary_capture_resource_is_deidentified(
    data_row_with_secondary: DataRow,
):
    """Regression test for the production crash on scan 501.

    A 'secondary' scan (rather than 'DICOM')
    should be deidentified successfully, with a well-formed output path.
    """
    utils.deidentify_dicom_files(
        data_row_with_secondary,
        score_threshold=0.5,
        spacy_model_name="en_core_web_md",
        destroy_pixels=True,
        use_transformers=False,
        dry_run=False,
    )

    keys = [
        k[0] if isinstance(k, tuple) else k
        for k in data_row_with_secondary.entries_dict.keys()
    ]

    # The secondary-capture scan should have a deidentified counterpart...
    deid_for_secondary = [
        k for k in keys
        if "patient_protocol" in k and "@deidentified" in k
    ]
    assert deid_for_secondary, (
        "Secondary-capture scan was not deidentified at all. "
        f"Entries: {keys}"
    )

    # ...and its path must be well-formed (no '/secondary' tail leaking through).
    for k in deid_for_secondary:
        assert "/secondary" not in k, (
            f"Malformed deidentified path: {k!r}. The '/secondary' resource "
            "label should not appear in the deidentified entry name."
        )

def test_ingest_anonymised_dicom(data_row: DataRow):
    n_scans_before = utils._count_dicom_files(data_row, resource_path=None)
    assert n_scans_before == 6
    utils.deidentify_dicom_files(data_row,
                                 score_threshold=0.5,
                                 spacy_model_name="en_core_web_md",
                                 destroy_pixels=True,
                                 use_transformers=False,
                                 dry_run=False)
    n_scans_after = utils._count_dicom_files(data_row, resource_path=None)
    dicom_files = utils._get_dicom_files(data_row)
    assert n_scans_after == 12
    for dicom_file in dicom_files[0:6]:
        assert np.any(dicom_file != 0)
        assert dicom_file.shape != (8, 8)
    for dicom_file in dicom_files[6:]:
        assert np.all(dicom_file == 0)  # Check if pixel data is destroyed
        assert dicom_file.shape == (8, 8)

    # Running again to ensure it does not duplicate entries.
    utils.deidentify_dicom_files(data_row,
                                 score_threshold=0.5,
                                 spacy_model_name="en_core_web_md",
                                 destroy_pixels=True,
                                 use_transformers=False,
                                 dry_run=False)
    n_scans_after = utils._count_dicom_files(data_row, resource_path=None)
    dicom_files = utils._get_dicom_files(data_row)
    assert n_scans_after == 12


def test_dry_run(data_row: DataRow):
    n_scans_before = utils._count_dicom_files(data_row, resource_path=None)
    assert n_scans_before == 6
    utils.deidentify_dicom_files(data_row,
                                 score_threshold=0.5,
                                 spacy_model_name="en_core_web_md",
                                 destroy_pixels=True,
                                 use_transformers=False,
                                 dry_run=True)
    n_scans_after = utils._count_dicom_files(data_row, resource_path=None)
    dicom_files = utils._get_dicom_files(data_row)
    assert n_scans_after == 6
    for dicom_file in dicom_files[0:6]:
        assert np.any(dicom_file != 0)
        assert dicom_file.shape != (8, 8)


def test_create_empty_entry(data_row: DataRow):
    # docker stop $(docker ps -aq); docker rm $(docker ps -aq)
    key = "fmap/DICOM"
    anonymised_key = key.replace("/DICOM", "@deidentified_empty")
    data_row.create_entry(anonymised_key, datatype=utils.DicomSeries)
    assert 0 == utils._count_dicom_files(data_row, resource_path=anonymised_key)


def _report_keys(data_row: DataRow) -> list:
    return [
        k[0] if isinstance(k, tuple) else k
        for k in data_row.entries_dict.keys()
        if (k[0] if isinstance(k, tuple) else k) == "deidentification_report@deidentified"
    ]


def test_pipeline_generates_report(data_row: DataRow):
    """deidentify_dicom_files uploads one session report, and re-runs re-use it."""
    utils.deidentify_dicom_files(data_row,
                                 score_threshold=0.5,
                                 spacy_model_name="en_core_web_md",
                                 destroy_pixels=True,
                                 use_transformers=False,
                                 dry_run=False)

    assert _report_keys(data_row) == ["deidentification_report@deidentified"]

    entry = data_row.entry("deidentification_report@deidentified")
    assert issubclass(entry.datatype, utils.File)
    # The report does not count as a DICOM scan.
    assert utils._count_dicom_files(
        data_row, resource_path="deidentification_report@deidentified"
    ) == 0

    contents = entry.item.read_contents()
    if isinstance(contents, bytes):
        contents = contents.decode("utf-8")
    assert "De-identification report" in contents
    assert "image(s) processed" in contents

    # Re-running the pipeline re-uses the report entry rather than duplicating it.
    utils.deidentify_dicom_files(data_row,
                                 score_threshold=0.5,
                                 spacy_model_name="en_core_web_md",
                                 destroy_pixels=True,
                                 use_transformers=False,
                                 dry_run=False)
    assert _report_keys(data_row) == ["deidentification_report@deidentified"]


def test_dry_run_generates_no_report(data_row: DataRow):
    """A dry-run anonymises nothing, so no report entry is created."""
    utils.deidentify_dicom_files(data_row,
                                 score_threshold=0.5,
                                 spacy_model_name="en_core_web_md",
                                 destroy_pixels=True,
                                 use_transformers=False,
                                 dry_run=True)
    assert _report_keys(data_row) == []


def test_save_html_report(data_row: DataRow):
    """An HTML report can be uploaded as its own (non-DICOM) entry and read back."""
    entry_name = "deidentification_report@deidentified"
    html_report = utils.build_html_report(
        [{"tag": "(0010, 0010)", "name": "Patient's Name"}],
        n_images=3,
        session_id=data_row.id,
        use_case="Standard",
    )

    n_dicom_before = utils._count_dicom_files(data_row, resource_path=None)

    entry = utils.save_html_report(data_row, html_report, entry_name=entry_name)

    # The entry exists on the row under the given name...
    keys = [
        k[0] if isinstance(k, tuple) else k
        for k in data_row.entries_dict.keys()
    ]
    assert entry_name in keys

    # It is a plain file, not a DICOM series.
    assert issubclass(entry.datatype, utils.File)
    assert not issubclass(entry.datatype, utils.DicomSeries)

    # It is not counted as a scan, and does not break counting the row.
    assert utils._count_dicom_files(data_row, resource_path=entry_name) == 0
    assert utils._count_dicom_files(data_row, resource_path=None) == n_dicom_before

    # Its contents round-trip back exactly.
    contents = entry.item.read_contents()
    if isinstance(contents, bytes):
        contents = contents.decode("utf-8")
    assert contents == html_report
    assert "Patient&#x27;s Name" in contents

    # Re-uploading re-uses the entry rather than duplicating it.
    utils.save_html_report(data_row, html_report, entry_name=entry_name)
    keys_after = [
        k[0] if isinstance(k, tuple) else k
        for k in data_row.entries_dict.keys()
    ]
    assert keys_after.count(entry_name) == 1


def test_debug_field_dump(data_row: DataRow) -> None:
    with data_row.frameset.store.connection:
        xsession = data_row.frameset.store.connection.session.projects[
            data_row.frameset.id
        ].experiments[data_row.id]
        xsession.fields["debug-dump"] = "test debug data"
        assert xsession.fields["debug-dump"] == "test debug data"
