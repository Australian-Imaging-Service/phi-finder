from pathlib import Path
from frametree.core.row import DataRow
import pydicom

from ml_anonymisation.dicom_tools import anonymise_dicom, utils


def test_data_row_access(tmp_path: Path, data_row: DataRow) -> None:
    assert len(data_row.entries) == 3
    assert isinstance(data_row, DataRow)
    
    #dicom_series = data_row.entry('fmap/DICOM').item
    #dicom_series.contents
    #dicom = dicom_series.contents[0]
    #dcm = pydicom.dcmread(dicom)
    #data_row.entries_dict.keys()  # dict_keys(['fmap/DICOM', 't1w/DICOM', 'dwi/DICOM'])
    #data_row.frameset.
    # dict_values([DataEntry(path='fmap/DICOM', 
    #               datatype=<class 'fileformats.medimage.dicom.DicomSeries'>, 
    #               row=DataRow(id=visit0group0member0, frequency=session), 
    #               uri='/data/projects/20250424153134basiced2d/subjects/Xnat4Tests_S00005/experiments/Xnat4Tests_E00005/scans/1/resources/DICOM', 
    #               order='1', quality=None), 
    #              DataEntry(path='t1w/DICOM', 
    #               datatype=<class 'fileformats.medimage.dicom.DicomSeries'>, 
    #               row=DataRow(id=visit0group0member0, frequency=session), 
    #               uri='/data/projects/20250424153134basiced2d/subjects/Xnat4Tests_S00005/experiments/Xnat4Tests_E00005/scans/2/resources/DICOM', 
    #               order='2', quality=None), 
    #              DataEntry(path='dwi/DICOM', 
    #               datatype=<class 'fileformats.medimage.dicom.DicomSeries'>, 
    #               row=DataRow(id=visit0group0member0, frequency=session), 
    #               uri='/data/projects/20250424153134basiced2d/subjects/Xnat4Tests_S00005/experiments/Xnat4Tests_E00005/scans/3/resources/DICOM', 
    #               order='3', quality=None)])


def test_utils_extract_dicom_from_data_row(tmp_path: Path, data_row: DataRow) -> None:
    path_and_dicom_dict = utils.extract_dicom_from_data_row(data_row)
    assert "path" in path_and_dicom_dict.keys()
    assert "dicom" in path_and_dicom_dict.keys()
    assert len(path_and_dicom_dict["path"]) == 30
    assert len(path_and_dicom_dict["dicom"]) == 30


def test_utils_anonymise_scans(tmp_path: Path, data_row: DataRow) -> None:
    path_and_dicom_dict = utils.anonymise_scans(data_row)
    assert "path" in path_and_dicom_dict.keys()
    assert "dicom" in path_and_dicom_dict.keys()
    assert "dicom_anonymised" in path_and_dicom_dict.keys()
    assert len(path_and_dicom_dict["path"]) == 30
    assert len(path_and_dicom_dict["dicom"]) == 30
    assert len(path_and_dicom_dict["dicom_anonymised"]) == 30
    for original, anonymised in zip(
        path_and_dicom_dict["dicom"], path_and_dicom_dict["dicom_anonymised"]
    ):
        assert type(original) == type(anonymised)


def test_rename_dicom_file():
    path1 = Path('/tmp/tmp2puqail2/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6.dcm')
    path2 = Path('/tmp/dcm/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6.dcm')
    path3 = Path("dcm/p/3.dcm")

    new_path1 = utils.rename_dicom_file(path1)
    new_path2 = utils.rename_dicom_file(path2)
    new_path3 = utils.rename_dicom_file(path3)

    assert new_path1 == Path('/tmp/tmp2puqail2/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6_deidentified.dcm')
    assert new_path2 == Path('/tmp/dcm/20250429170214basicfa14/subjects/Xnat4Tests_S00006/experiments/Xnat4Tests_E00006/scans/1/resources/DICOM/6_deidentified.dcm')
    assert new_path3 == Path("dcm/p/3_deidentified.dcm")


def test_put_dicom_back(data_row: DataRow):
    path_and_dicom_dict = utils.extract_dicom_from_data_row(data_row)
    path_and_dicom_dict["path_anonymised"] = []

    for path in path_and_dicom_dict["path"]:
        path_and_dicom_dict["path_anonymised"].append(utils.rename_dicom_file(path))
    assert len(path_and_dicom_dict["path_anonymised"]) == 30
