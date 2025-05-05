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
        anonymised_session_entry = data_row.create_entry(anonymised_session_key, datatype=DicomSeries)
        try:
            dicom_series = data_row.entry(session_key).item
        except:
            print(f"Nothing found in data row {session_key}.")
            continue
        anonymised_session_entry.item = DicomSeries(dicom_series.contents)
        for dicom in data_row.entry(anonymised_session_key).item.contents:
            dcm = pydicom.dcmread(dicom)
            anonymised_dcm = anonymise_dicom.anonymise_image(dcm)
            anonymised_dcm.save_as(dicom)
    return data_row


def _count_dicom_files(data_row, session_key=None) -> int:
    """Counts the number of dicom files in a data row.
    If session_key is None, it counts the number of dicom files in all sessions.
    """
    def _list(session_key):
        try:
            dicom_series = data_row.entry(session_key).item
        except:
            print(f"Nothing found in data row {session_key}.")
            return 0

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
    