from pathlib import Path
from frametree.core.row import DataRow
from fileformats.medimage.dicom import DicomSeries
import pydicom

from phi_finder.dicom_tools import anonymise_dicom


def deidentify_dicom_files(data_row: DataRow,
                           score_threshold: float=0.5,
                           destroy_pixels: bool=True,
                           use_transformers: bool=False) -> None:
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

    score_threshold : float, optional (default 0.5)
        The score threshold for entity recognition. Entities with a score below this
        threshold will not be considered for anonymisation.
    
    destroy_pixels : bool, optional (default True)
        If True, the pixel data in the DICOM files will be a small black matrix.

    use_transformers : bool, optional (default False)
        If True, transformers will be used for anonymisation on top of Presidio's output.

    Returns
    -------
    data_row : None
    """
    entries = list(data_row.entries_dict.items())
    for resource_path, entry in entries:
        # 0. Check if the entry is a DICOM series and not a derivative.
        if entry.datatype != DicomSeries:
            print(f"Skipping {resource_path} as it is not a DICOM series.")
            continue
        if entry.is_derivative:
            print(f"Skipping {resource_path} as it is a derivative.")
            continue
        anonymised_resource_path = resource_path.replace("/DICOM", "@deidentified")
        if anonymised_resource_path in [x[0] for x in entries]:  # x: (name: str, entry: DataEntry)
            print(f"Skipping {resource_path} as it is already anonymised.")
            continue

        print(f"De-identifying {resource_path} to {anonymised_resource_path}.")
        # 1. Downloading the files from the original scan entry.
        dicom_series = entry.item

        # 2. Anonymising those files.
        tmps_paths = []
        for i, dicom in enumerate(dicom_series.contents):
            dcm = pydicom.dcmread(dicom)
            anonymised_dcm = anonymise_dicom.anonymise_image(dcm,
                                                             score_threshold=score_threshold,
                                                             use_transformers=use_transformers)
            if destroy_pixels:
               anonymised_dcm = anonymise_dicom.destroy_pixels(anonymised_dcm)
            tmp_path = Path(f"anonymised{i}-tmp_{dicom.stem}.dcm")
            anonymised_dcm.save_as(tmp_path)
            tmps_paths.append(tmp_path)

        # 3. Creating the deidentified entry.
        anonymised_session_entry = data_row.create_entry(
            anonymised_resource_path, datatype=DicomSeries
        )

        # 4. Creating a new DicomSeries object from the anonymised files.
        anonymised_dcm_series = DicomSeries(tmps_paths)

        # 5. Uploading the anonymised files from the temp dir.
        anonymised_session_entry.item = anonymised_dcm_series
    return None


def _get_dicom_files(data_row: DataRow) -> list:
    """Returns a list of DICOM files in a data row.
    If session_key is None, it returns the DICOM files in all sessions.

    Parameters
    ----------
    data_row : DataRow
        The data row containing the DICOM files.

    Returns
    -------
    list[pixel_array]
        A list of pixel arrays of the DICOM images.
    """
    def _get_dicom_in_session(session_key: str | None):
        try:
            dicom_series = data_row.entry(session_key).item
            paths = dicom_series.contents
            pixel_arrays = [pydicom.dcmread(path).pixel_array for path in paths]
        except:
            print(f"Nothing found in data row {session_key}.")
            return 0
        return pixel_arrays

    resource_paths = list(data_row.entries_dict.keys())
    dicom_files = []
    for resource_path in resource_paths:
        dicom_files.extend(_get_dicom_in_session(resource_path))
    return dicom_files


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
        except:
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
