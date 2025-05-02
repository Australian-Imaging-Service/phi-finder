from pathlib import Path
from frametree.core.row import DataRow
import pydicom

from ml_anonymisation.dicom_tools import anonymise_dicom, utils


def test_data_row_access(tmp_path: Path, data_row: DataRow) -> None:
    assert len(data_row.entries) == 3
    assert isinstance(data_row, DataRow)


def test_create_new_dicom_filepath():
    path1 = Path('/tmp/tmp2puqail2/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6.dcm')
    path2 = Path('/tmp/dcm/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6.dcm')
    path3 = Path("dcm/p/3.dcm")

    new_path1 = utils.create_new_dicom_filepath(path1)
    new_path2 = utils.create_new_dicom_filepath(path2)
    new_path3 = utils.create_new_dicom_filepath(path3)

    assert new_path1 == Path('/tmp/tmp2puqail2/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6_deidentified.dcm')
    assert new_path2 == Path('/tmp/dcm/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6_deidentified.dcm')
    assert new_path3 == Path("dcm/p/3_deidentified.dcm")


def test_ingest_anonymised_dicom(data_row: DataRow):
    n_scans_before = utils._count_dicom_files(data_row, session_key=None)
    assert n_scans_before == 6
    data_row = utils.deidentify_dicom_files(data_row)
    n_scans_after = utils._count_dicom_files(data_row, session_key=None)
    assert n_scans_after == 12
