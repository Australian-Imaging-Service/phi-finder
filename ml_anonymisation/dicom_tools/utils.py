from pathlib import Path
from frametree.core.row import DataRow
import pydicom

from ml_anonymisation.dicom_tools import anonymise_dicom


def extract_dicom_from_data_row(data_row: DataRow) -> dict:
    """
    Extracts the dicom series from the data row.
    A dict with the keys 'path' and 'dicom' is returned.
    """
    keys = data_row.entries_dict.keys()
    path_and_dicom_dict = {"path": [], "dicom": []}
    for key in keys:  # e.g. ['fmap/DICOM', 't1w/DICOM', 'dwi/DICOM']
        dicom_series = data_row.entry(key).item
        for dicom in dicom_series.contents:
            dcm = pydicom.dcmread(dicom)
            path = dicom.absolute()
            path_and_dicom_dict["path"].append(path)
            path_and_dicom_dict["dicom"].append(dcm)
    return path_and_dicom_dict


def anonymise_scans(data_row: DataRow):
    """
    Anonymises the dicom series in the data row.
    """
    path2dicom = extract_dicom_from_data_row(data_row)
    path2dicom["dicom_anonymised"] = []
    for path, dicom in zip(path2dicom["path"], path2dicom["dicom"]):
        anonymised_dicom = anonymise_dicom.anonymise_image(dicom)
        path2dicom["dicom_anonymised"].append(anonymised_dicom)
    return path2dicom
    