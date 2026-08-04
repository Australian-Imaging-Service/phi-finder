import gc
import tempfile
from pathlib import Path

import pydicom
from fileformats.medimage.dicom import DicomSeries
from frametree.core.row import DataRow
from gliner.model import UniEncoderSpanGLiNER
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_image_redactor import DicomImageRedactorEngine, ImageAnalyzerEngine, ContrastSegmentedImageEnhancer

from phi_finder.dicom_tools import anonymise_dicom, html_report, ps3_15


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


def _build_engines(use_case: str,
                   score_threshold: float,
                   spacy_model_name: str,
                   destroy_pixels: bool,
                   use_transformers: bool) -> tuple[AnalyzerEngine | None,
                                                    AnonymizerEngine | None,
                                                    DicomImageRedactorEngine | None,
                                                    UniEncoderSpanGLiNER | None]:
    """Builds the engines deidentify_dicom_files needs for a given use case.

    In the PS3.15 use cases the standard headers are handled by the basic
    profile. The NER engines (Presidio and GLiNER) are only needed when the 
    full NER pipeline runs on the headers, or for the "..._scan_private" variants.
    Presidio also redacts burned-in pixel PHI (if destroy_pixels=False).

    Parameters
    ----------
    use_case : str
        The de-identification use case (see deidentify_dicom_files).

    score_threshold : float
        The score threshold for entity recognition.

    spacy_model_name : str
        The name of the SpaCy model to use for NLP processing.

    destroy_pixels : bool
        If True, pixel data is destroyed, so no image redactor is needed.

    use_transformers : bool
        If True, GLiNER is used on top of Presidio wherever the NER pipeline runs.

    Returns
    -------
    tuple
        (analyser, anonymizer, image_redactor, gliner_pii), each None when the
        use case does not need it.
    """
    ps3_15_mode = ps3_15.is_ps3_15_use_case(use_case)
    ner_needed = not ps3_15_mode or ps3_15.scan_private_headers(use_case)
    if ner_needed or destroy_pixels is False:
        analyser = anonymise_dicom._build_presidio_analyser(score_threshold, spacy_model_name)
    else:
        analyser = None
    anonymizer = AnonymizerEngine() if ner_needed else None
    if destroy_pixels is False:
        image_redactor = (DicomImageRedactorEngine(
                image_analyzer_engine=ImageAnalyzerEngine(analyzer_engine=analyser,
                                                          image_preprocessor=ContrastSegmentedImageEnhancer()))
        )
    else:
        image_redactor = None
    # GLiNER is built whenever the caller asked for it: even a PS3.15 use case
    # that leaves the standard headers to the profile runs the NER pipeline over
    # the free-text attributes the profile has no action for (e.g. SR Text
    # Value), and those are exactly the long values GLiNER is there to catch.
    gliner_pii = anonymise_dicom._build_transformer() if use_transformers else None
    return analyser, anonymizer, image_redactor, gliner_pii


def deidentify_dicom_files(data_row: DataRow,
                           score_threshold: float=0.5,
                           spacy_model_name: str="en_core_web_md",
                           destroy_pixels: bool=True,
                           use_transformers: bool=False,
                           dry_run: bool=False,
                           use_case: str='dicom_retain_patient_scan_private') -> None:
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

    spacy_model_name : str, optional (default "en_core_web_md")
        The name of the SpaCy model to use for NLP processing.
        Other options include "en_core_web_sm" and "en_core_web_lg".
    
    destroy_pixels : bool, optional (default True)
        If True, the pixel data in the DICOM files will be a small black matrix.

    use_transformers : bool, optional (default False)
        If True, transformers will be used for anonymisation on top of Presidio's output.

    dry_run : bool, optional (default False)
        If True, the function will not perform any changes, only log the actions that would be taken.
        Note that original DICOM files will still be loaded.

    use_case : str, optional (default 'dicom_retain_patient_scan_private')
        * PS3.15 (alias 'dicom_default'): headers are de-identified with the
        DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile;
        Presidio and GLiNER are not used on the headers.
        * PS3.15_Rtn. Pat. (alias 'dicom_retain_patient'): as PS3.15, plus the
        Retain Patient Characteristics Option, so patient characteristics
        (age, sex, weight, ...) are kept.
        * 'dicom_default_scan_private' / 'dicom_retain_patient_scan_private': as
        the matching PS3.15 variant for the standard headers, but private
        attributes are kept and scanned with the Presidio/GLiNER pipeline
        instead of being removed.
        * Any other (e.g. 'Standard', 'NER Only'): headers are dealt with the
        Presidio NER pipeline (plus GLiNER if use_transformers).

    Returns
    -------
    None : None
        The function does not return anything.

    """
    _log_session(data_row, "debug-dump0", "Pipeline started")

    analyser, anonymizer, image_redactor, gliner_pii = _build_engines(
        use_case, score_threshold, spacy_model_name, destroy_pixels,
        use_transformers,
    )

    # Accumulated across every scan/slice in the session to build one report.
    report_headers = []
    note_diffs = []
    n_images = 0

    entries = list(data_row.entries_dict.items())
    for resource_path_key_order, entry in entries:
        gc.collect()
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
        #anonymised_resource_path = str(order_key) + '_' + resource_path.replace("/DICOM", "@deidentified")
        scan_name, sep, _resource_label = resource_path.rpartition("/")
        if not sep:
            # No '/' in path: treat the whole string as the scan name
            scan_name = resource_path
        anonymised_resource_path = f"{order_key}_{scan_name}@deidentified"

        print(f"De-identifying {resource_path} to {anonymised_resource_path}.")
        _log_session(data_row, "debug-dump2", f"De-identifying {resource_path} to {anonymised_resource_path}.")

        # 1. Downloading the files from the original scan entry.
        try:
            dicom_series = entry.item
        except AssertionError as e:
            print(f"AssertionError occurred while downloading files from {resource_path}: {e}")
            _log_session(data_row, "debug-dump3", f"AssertionError occurred while downloading files from {resource_path}: {e}")
            continue
        _log_session(data_row, "debug-dump3", f"Files from the original scan entry were downloaded.")

        # 2. Anonymising those files. The temp dir is unique per entry and
        # run, so concurrent pipelines cannot overwrite each other's files,
        # and it is removed once the upload has completed.
        with tempfile.TemporaryDirectory(prefix="phi-finder-") as tmp_dir:
            tmps_paths = []
            for i, dicom in enumerate(dicom_series.contents):
                gc.collect()
                dcm = pydicom.dcmread(dicom)
                if dry_run:
                    continue
                # Snapshot the long free-text fields before anonymise_image
                # mutates the dataset in place, so we can diff them afterwards.
                note_snapshot = html_report.snapshot_long_text(dcm)
                anonymised_dcm = anonymise_dicom.anonymise_image(dcm,
                                                                 analyser=analyser,
                                                                 anonymizer=anonymizer,
                                                                 image_redactor=image_redactor,
                                                                 score_threshold=score_threshold,
                                                                 gliner_pii=gliner_pii,
                                                                 use_case=use_case)
                report_headers.extend(html_report.read_flagged_headers(anonymised_dcm))
                note_diffs.extend(html_report.collect_note_diffs(note_snapshot, anonymised_dcm))
                n_images += 1
                if destroy_pixels:
                    anonymised_dcm = anonymise_dicom.destroy_pixels(anonymised_dcm)
                tmp_path = Path(tmp_dir) / f"anonymised{i}-tmp_{dicom.stem}.dcm"
                anonymised_dcm.save_as(tmp_path)
                tmps_paths.append(tmp_path)

            if dry_run:
                _log_session(data_row, "debug-dump4", f"Files anonymised (dry-run).")
            else:
                _log_session(data_row, "debug-dump4", f"Files anonymised.")

            # 3. Creating the deidentified entry if necessary.
            entries_names = [x[0][0] for x in entries]  # x: ((name: str, order_key: str), entry: DataEntry)
            if dry_run:
                _log_session(data_row, "debug-dump6", f"Deidentified files uploaded (dry-run).")
                continue

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

    # 6. Building and uploading de-identification report for the whole session.
    if not dry_run:
        report_html = html_report.build_html_report(
            report_headers, n_images,
            session_id=data_row.id, use_case=use_case,
            note_diffs=note_diffs,
        )
        html_report.save_html_report(data_row, report_html)
        _log_session(data_row, "debug-dump7", "De-identification report uploaded.")
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
            entry = data_row.entry(session_key)
            # Skip non-DICOM entries (e.g. an HTML report); they have no pixels.
            if not issubclass(entry.datatype, DicomSeries):
                return []
            dicom_series = entry.item
            paths = dicom_series.contents
            pixel_arrays = [pydicom.dcmread(path).pixel_array for path in paths]
        except:
            print(f"Nothing found in data row {session_key}.")
            return []
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
            entry = data_row.entry(session_key)
            # Skip non-DICOM entries (e.g. an HTML report), whose contents
            # cannot be enumerated as a DICOM series and are not scans to count.
            if not issubclass(entry.datatype, DicomSeries):
                return 0
            dicom_series = entry.item
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
