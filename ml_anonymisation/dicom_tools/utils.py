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
    Anonymises all dicom in the data row.

    A dict with the keys 'path', 'dicom' and 'dicom_anonymised' is returned.
    """
    path_and_dicom_dict = extract_dicom_from_data_row(data_row)
    path_and_dicom_dict["dicom_anonymised"] = []
    for path, dicom in zip(path_and_dicom_dict["path"], path_and_dicom_dict["dicom"]):
        anonymised_dicom = anonymise_dicom.anonymise_image(dicom)
        path_and_dicom_dict["dicom_anonymised"].append(anonymised_dicom)
    return path_and_dicom_dict


def rename_dicom_file(dicom_path: Path) -> Path:
    """
    Renames the dicom file to a new name.
    The new name is the same as the original name, but ending with the suffix _deidentified.

    Example: "dicom_path/1.dcm" -> "dicom_path/1_deidentified.dcm"
    """
    file_stem = dicom_path.stem  # e.g. 1
    file_extension = dicom_path.suffix  # e.g. .dcm
    new_file_name = f"{file_stem}_deidentified{file_extension}"
    new_dicom_path = dicom_path.parent / new_file_name
    return new_dicom_path
