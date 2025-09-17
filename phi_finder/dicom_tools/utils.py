from pathlib import Path
from frametree.core.row import DataRow
from fileformats.medimage.dicom import DicomSeries
import pydicom

from phi_finder.dicom_tools import anonymise_dicom


def _log_session(data_row: DataRow, key: str, message: str) -> None:
    """Logs a message to the session's debug-dump field.

    Parameters
    ----------
    data_row : DataRow
        The data row containing the session to log the message to.

    key : str
        The key of the field to log the message to.

    message : str
        The message to log.

    Returns
    -------
    None : None
        The function does not return anything.
    """
    with data_row.frameset.store.connection:
        xlogin = data_row.frameset.store.connection.session
        xproject = xlogin.projects[data_row.frameset.id]
        xsession = xproject.experiments[data_row.id]
        xsession.fields[key] = message
    return None


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
    None : None
        The function does not return anything.

    """
    _log_session(data_row, "debug-dump0", "Pipeline started")

    entries = list(data_row.entries_dict.items())
    for resource_path_key_order, entry in entries:
        resource_path = resource_path_key_order[0]
        order_key = resource_path_key_order[1]
        # 0. Check if the entry is a DICOM series and not a derivative.
        if entry.datatype != DicomSeries:
            print(f"Skipping {resource_path} as it is not a DICOM series.")
            _log_session(data_row, "debug-dump1", f"Skipping {resource_path} as it is not a DICOM series.")
            continue
        if entry.is_derivative:
            print(f"Skipping {resource_path} as it is a derivative.")
            _log_session(data_row, "debug-dump1", f"Skipping {resource_path} as it is a derivative.")
            continue
        anonymised_resource_path = str(order_key) + '_' + resource_path.replace("/DICOM", "@deidentified")

        print(f"De-identifying {resource_path} to {anonymised_resource_path}.")
        _log_session(data_row, "debug-dump2", f"De-identifying {resource_path} to {anonymised_resource_path}.")

        # 1. Downloading the files from the original scan entry.
        dicom_series = entry.item
        _log_session(data_row, "debug-dump3", f"Files from the original scan entry were downloaded.")

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
        _log_session(data_row, "debug-dump4", f"Files anonymised.")

        # 3. Creating the deidentified entry if necessary.
        entries_names = [x[0][0] for x in entries]  # x: ((name: str, order_key: str), entry: DataEntry)
        if anonymised_resource_path in entries_names:
            print(f"Re-using {anonymised_resource_path} that already exists.")
            _log_session(data_row, "debug-dump5", f"Re-using {anonymised_resource_path} that already exists.")
            index = entries_names.index(anonymised_resource_path)
            anonymised_session_entry = entries[index][1]
        else:
            anonymised_session_entry = data_row.create_entry(
                anonymised_resource_path, datatype=DicomSeries, order_key=order_key
            )
            _log_session(data_row, "debug-dump5", f"Deidentified entry created.")

        # 4. Creating a new DicomSeries object from the anonymised files.
        anonymised_dcm_series = DicomSeries(tmps_paths)

        # 5. Uploading the anonymised files from the temp dir.
        anonymised_session_entry.item = anonymised_dcm_series
        _log_session(data_row, "debug-dump6", f"Deidentified files uploaded.")

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
    resource_paths = [x[0] if isinstance(x, tuple) else x for x in resource_paths]
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
        )  # Copy, not reference, of the keys, e.g. [('fmap/DICOM', '1'), ('t1w/DICOM', '1'), ('dwi/DICOM', '1')]
        n_scans = 0
        session_keys = [x[0] if isinstance(x, tuple) else x for x in session_keys]
        for resource_path in session_keys:
            n_scans += _count_dicom_in_session(resource_path)
        return n_scans
    else:
        return _count_dicom_in_session(resource_path)
