from pathlib import Path
from frametree.core.row import DataRow
from fileformats.medimage.dicom import DicomSeries
import pydicom

from ml_anonymisation.dicom_tools import anonymise_dicom


def deidentify_dicom_files(data_row: DataRow) -> None:
    """Main function to deidentify dicom files in a data row.
    Loop over sessions in data_row.entries
        Create a new session entry
        Loop over images in session
            Anonymise the image
            Create a new name for the anonymised image
            Save the anonymised image into the new session entry
        Add the paths of the anonymised images to the new session entry
    """
    session_keys = list(data_row.entries_dict.keys())  # Copy, not reference, of the keys, e.g. ['fmap/DICOM', 't1w/DICOM', 'dwi/DICOM']
    for session_key in session_keys:
        anonymised_session_key = session_key.replace("/DICOM", "@deidentified")
        dicom_series = data_row.entry(session_key).item
        anonymised_dcms = []
        anonymised_dicom_paths = []
        anonymised_session_entry = data_row.create_entry(anonymised_session_key, datatype=DicomSeries)
        for dicom in dicom_series.contents:
            dcm = pydicom.dcmread(dicom)
            anonymised_dcm = anonymise_dicom.anonymise_image(dcm)
            anonymised_dicom_path = create_new_dicom_filepath(dicom.absolute())
            anonymised_dicom_paths.append(anonymised_dicom_path)
            anonymised_dcm.save_as(anonymised_dicom_path)
        anonymised_session_entry.item = DicomSeries(anonymised_dicom_paths)
    return data_row


def _list_dicom_files(data_row, session_key=None) -> int:
    def _list(session_key):
        dicom_series = data_row.entry(session_key).item
        for i, dicom in enumerate(dicom_series.contents):
            print(i, dicom.absolute())
        return i+1  # Returning the number of dicom files in the series.
    if not session_key:
        session_keys = list(data_row.entries_dict.keys())  # Copy, not reference, of the keys, e.g. ['fmap/DICOM', 't1w/DICOM', 'dwi/DICOM']
        n_scans = 0
        for session_key in session_keys:
            n_scans += _list(session_key)
        return n_scans
    else:
        return _list(session_key)


def create_new_dicom_filepath(dicom_path: Path) -> Path:
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
