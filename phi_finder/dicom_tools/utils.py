from pathlib import Path
from frametree.core.row import DataRow
from fileformats.medimage.dicom import DicomSeries
import pydicom

from phi_finder.dicom_tools import anonymise_dicom


def deidentify_dicom_files(data_row: DataRow) -> DataRow:
    """Main function to deidentify dicom files in a data row.
        1. Download the files from the original scan entry fmap/DICOM
        2. Anonymise those files and store the anonymised files in a temp dir
        3. Create the deidentified entry using deid_entry = create_entry(...)
        4. Create  a new DicomSeries object from the anonymised files dicom_series = DicomSeries('anonymised-tmp/1.dcm', ...)
        5. Upload the anonymised files from the temp dir with deid_entry.item = dicom_series
    
    Parameters
    ----------
    data_row : DataRow
        The data row containing the DICOM files to be deidentified.

    Returns
    -------
    data_row : DataRow
        The DataRow containing the original and anonymised DICOM files.
    """
    session_keys = list(data_row.entries_dict.keys())  # Copy, not reference, of the keys, e.g. ['fmap/DICOM', 't1w/DICOM', 'dwi/DICOM']
    for session_key in session_keys:
        # 1. Downloading the files from the original scan entry.
        try:
            dicom_series = data_row.entry(session_key).item
        except:
            print(f"Nothing found in data row {session_key}.")
            continue

        # 2. Anonymising those files.
        tmps_paths = []
        for i, dicom in enumerate(dicom_series.contents):
            dcm = pydicom.dcmread(dicom)
            anonymised_dcm = anonymise_dicom.anonymise_image(dcm)
            tmp_path = Path(f"anonymised{i}-tmp_{dicom.stem}.dcm")
            anonymised_dcm.save_as(tmp_path)
            tmps_paths.append(tmp_path)

        # 3. Creating the deidentified entry.
        anonymised_session_key = session_key.replace("/DICOM", "@deidentified")
        anonymised_session_entry = data_row.create_entry(anonymised_session_key, datatype=DicomSeries)

        # 4. Creating a new DicomSeries object from the anonymised files.
        anonymised_dcm_series = DicomSeries(tmps_paths)

        # 5. Uploading the anonymised files from the temp dir.
        anonymised_session_entry.item = anonymised_dcm_series
    return data_row


def _count_dicom_files(data_row: DataRow, session_key=None) -> int:
    """Counts the number of dicom files in a data row.
    If session_key is None, it counts the number of dicom files in all sessions.

    Parameters
    ----------
    data_row : DataRow
        The data row containing the DICOM files.
    
    session_key : str, optional
        The session key for which to count the DICOM files. If None, counts for all sessions.

    Returns
    -------
    int
        The number of DICOM files in the specified session or in all sessions if session_key is None.
    """
    def _count_dicom_in_session(session_key: str) -> int:
        """Helper function to list DICOM files in a specific session.

        Args:
            session_key (str): The session key to list DICOM files for.

        Returns:
            int: The number of DICOM files in the specified session.
        """
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
            n_scans += _count_dicom_in_session(session_key)
        return n_scans
    else:
        return _count_dicom_in_session(session_key)
    