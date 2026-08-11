"""Aggregates several phi-finder HTML de-identification reports into one.

"""

import html
import tempfile
from pathlib import Path
from typing import Optional

from fileformats.generic import File
from frametree.core.row import DataRow


_DEFAULT_FILENAME = "aggregated_report.html"

# Resource path each ``deidentify_dicom_files`` run attaches its per-session report
# under (see ``phi_finder.dicom_tools.html_report.save_html_report``).
_DEFAULT_REPORT_ENTRY = "deidentification_report@deidentified"
# Resource path the dataset-level aggregated report is written back under.
_DEFAULT_OUTPUT_ENTRY = "aggregated_deidentification_report@deidentified"


def _combine_reports(documents: "list[str]",
                    labels: "Optional[list[str]]" = None) -> str:
    """Combines loaded reports into a single HTML document.

    Placeholder logic: each report is embedded whole, in the order given, under
    a heading naming the source it came from. TODO.

    Parameters
    ----------
    documents : list of str
        The report documents, as returned by ``load_reports``.
    labels : list of str, optional
        Section headings for each document (e.g. the source file name or the
        session id). Must be the same length as ``documents`` when given;
        defaults to ``"Report 1"``, ``"Report 2"``, ...

    Returns
    -------
    str
        A self-contained HTML document holding every input report.
    """
    labels = labels if labels else [
        f"Report {i + 1}" for i in range(len(documents))
    ]
    sections = "\n".join(
        f"    <section>\n      <h2>{html.escape(label)}</h2>\n{document}\n"
        "    </section>"
        for label, document in zip(labels, documents)
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aggregated de-identification report</title>
</head>
<body>
  <main>
    <h1>Aggregated de-identification report</h1>
    <p>{len(documents)} report(s) aggregated.</p>
{sections}
  </main>
</body>
</html>
"""


def _entry_names(row: DataRow) -> "list[str]":
    """Returns the resource-path names of a row's entries.

    ``entries_dict`` is keyed by ``(name, order_key)`` tuples; this collapses
    those to just the names so callers can test membership by resource path.
    """
    return [k[0] if isinstance(k, tuple) else k for k in row.entries_dict.keys()]


def aggregate_reports(
        data_row: DataRow,
        report_entry_name: str = _DEFAULT_REPORT_ENTRY,
        output_entry_name: str = _DEFAULT_OUTPUT_ENTRY) -> None:
    """Aggregates every session's de-identification report into one dataset report.

    This is the pydra2app command entrypoint. It runs on the ``medimage/constant``
    (dataset root) row and, like ``deidentify_dicom_files``, does its own data
    access through the frametree row API rather than declaring typed file inputs:
    frametree cannot serialise a collection-typed task field (e.g. ``list[Html]``),
    so the fan-in over sessions is done here instead.

    It walks every session row, collects the HTML report that
    ``deidentify_dicom_files`` attached under ``report_entry_name``, combines them,
    and uploads the aggregated document back onto the constant row under
    ``output_entry_name``. Sessions without a report are skipped.

    Parameters
    ----------
    data_row : DataRow
        The ``medimage/constant`` (dataset root) row the command operates on.
    report_entry_name : str, optional
        Resource path of the per-session report entry to collect. Defaults to
        the path written by ``save_html_report``.
    output_entry_name : str, optional
        Resource path to write the aggregated report to on the constant row.

    Returns
    -------
    None : None
        The aggregated report is uploaded to the data row; nothing is returned.
    """
    documents = []
    labels = []
    for session in data_row.frameset.rows("session"):
        if report_entry_name not in _entry_names(session):
            continue
        # Assignment access (``.item``) downloads the report file locally.
        report_file = session.entry(report_entry_name).item
        documents.append(Path(report_file).read_text(encoding="utf-8"))
        labels.append(session.id)

    aggregated = _combine_reports(documents, labels=labels)

    with tempfile.TemporaryDirectory(prefix="phi-finder-aggregate-") as tmp_dir:
        destination = Path(tmp_dir) / _DEFAULT_FILENAME
        destination.write_text(aggregated, encoding="utf-8")

        if output_entry_name in _entry_names(data_row):
            entry = data_row.entry(output_entry_name)
        else:
            entry = data_row.create_entry(output_entry_name, datatype=File)
        # Assignment uploads the file while the temp dir is still alive.
        entry.item = File(destination)
    return None
