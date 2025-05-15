from pathlib import Path
from frametree.core.row import DataRow
from fileformats.medimage.dicom import DicomSeries
import pydicom

from phi_finder.dicom_tools import anonymise_dicom


def deidentify_dicom_files(data_row: DataRow) -> None:
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
    for resource_path, entry in data_row.entries_dict.items():
        # 0. Check if the entry is a DICOM series and not a derivative.
        if entry.datatype != DicomSeries:
            print(f"Skipping {resource_path} as it is not a DICOM series.")
            continue
        if entry.is_derivative:
            print(f"Skipping {resource_path} as it is a derivative.")
            continue

        # 1. Downloading the files from the original scan entry.
        dicom_series = entry.item

        # 2. Anonymising those files.
        tmps_paths = []
        for i, dicom in enumerate(dicom_series.contents):
            dcm = pydicom.dcmread(dicom)
            anonymised_dcm = anonymise_dicom.anonymise_image(dcm)
            tmp_path = Path(f"anonymised{i}-tmp_{dicom.stem}.dcm")
            anonymised_dcm.save_as(tmp_path)
            tmps_paths.append(tmp_path)

        # 3. Creating the deidentified entry.
        anonymised_resource_path = resource_path.replace("/DICOM", "@deidentified")
        anonymised_session_entry = data_row.create_entry(
            anonymised_resource_path, datatype=DicomSeries
        )

        # 4. Creating a new DicomSeries object from the anonymised files.
        anonymised_dcm_series = DicomSeries(tmps_paths)

        # 5. Uploading the anonymised files from the temp dir.
        anonymised_session_entry.item = anonymised_dcm_series
    return None


def _count_dicom_files(data_row: DataRow, resource_path: str | None = None) -> int:
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

    def _count_dicom_in_session(session_key: str | None) -> int:
        """Helper function to list DICOM files in a specific session.

        Args:
            session_key (str): The session key to list DICOM files for.

        Returns:
            int: The number of DICOM files in the specified session.
        """
        try:
            dicom_series = data_row.entry(session_key).item
        except KeyError:
            print(f"Nothing found in data row {session_key}.")
            return 0

        for i, dicom in enumerate(dicom_series.contents):
            print(i, dicom.absolute())
        return i + 1  # Returning the number of dicom files in the series.

    if not resource_path:
        session_keys = list(
            data_row.entries_dict.keys()
        )  # Copy, not reference, of the keys, e.g. ['fmap/DICOM', 't1w/DICOM', 'dwi/DICOM']
        n_scans = 0
        for resource_path in session_keys:
            n_scans += _count_dicom_in_session(resource_path)
        return n_scans
    else:
        return _count_dicom_in_session(resource_path)
