import numpy as np
from pathlib import Path
from frametree.core.row import DataRow

from phi_finder.dicom_tools import utils


def test_data_row_access(tmp_path: Path, data_row: DataRow) -> None:
    assert len(data_row.entries) == 3
    assert isinstance(data_row, DataRow)


def test_ingest_anonymised_dicom(data_row: DataRow):
    n_scans_before = utils._count_dicom_files(data_row, resource_path=None)
    assert n_scans_before == 6
    utils.deidentify_dicom_files(data_row, destroy_pixels=True)
    n_scans_after = utils._count_dicom_files(data_row, resource_path=None)
    dicom_files = utils._get_dicom_files(data_row)
    assert n_scans_after == 12
    for dicom_file in dicom_files[0:6]:
        assert np.any(dicom_file != 0)
    for dicom_file in dicom_files[6:]:
        assert np.all(dicom_file == 0)  # Check if pixel data is destroyed


def test_create_empty_entry(data_row: DataRow):
    # docker stop $(docker ps -aq); docker rm $(docker ps -aq)
    key = "fmap/DICOM"
    anonymised_key = key.replace("/DICOM", "@deidentified_empty")
    data_row.create_entry(anonymised_key, datatype=utils.DicomSeries)
    assert 0 == utils._count_dicom_files(data_row, resource_path=anonymised_key)
