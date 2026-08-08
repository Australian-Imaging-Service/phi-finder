"""Aggregates several phi-finder HTML de-identification reports into one.

The aggregation logic itself is a placeholder: the reports are currently
concatenated in the order they were given, wrapped in a single document. Only
the interface -- paths in, a ``fileformats`` ``Html`` file out -- is settled.
"""

import html
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

from fileformats.text.unicode import Html

# Where an aggregated report is written when the caller names no destination.
_DEFAULT_FILENAME = "aggregated_report.html"


def load_reports(report_paths: Iterable[Union[str, Path]]) -> "list[str]":
    """Reads each report file into memory.

    Parameters
    ----------
    report_paths : iterable of str or pathlib.Path
        Paths to the HTML reports to load, e.g. as written by
        ``phi_finder.dicom_tools.html_report.build_html_report``.

    Returns
    -------
    list of str
        The documents' text, in the order the paths were given.

    Raises
    ------
    FileNotFoundError
        If any of the paths does not exist.
    """
    documents = []
    for report_path in report_paths:
        path = Path(report_path)
        if not path.is_file():
            raise FileNotFoundError(f"No such report file: {path}")
        documents.append(path.read_text(encoding="utf-8"))
    return documents


def combine_reports(documents: "list[str]",
                    sources: "Optional[list[Path]]" = None) -> str:
    """Combines loaded reports into a single HTML document.

    Placeholder logic: each report is embedded whole, in the order given, under
    a heading naming the file it came from. How the reports should really be
    merged (shared summary, de-duplicated fields, per-session sections) is not
    decided yet.

    Parameters
    ----------
    documents : list of str
        The report documents, as returned by ``load_reports``.
    sources : list of pathlib.Path, optional
        The paths the documents came from, used to label each section. Must be
        the same length as ``documents`` when given.

    Returns
    -------
    str
        A self-contained HTML document holding every input report.
    """
    labels = [p.name for p in sources] if sources else [
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


def aggregate_reports(report_paths: Iterable[Union[str, Path]],
                      output_path: "Optional[Union[str, Path]]" = None) -> Html:
    """Aggregates HTML de-identification reports into a single ``Html`` file.

    Parameters
    ----------
    report_paths : iterable of str or pathlib.Path
        Paths to the HTML reports to aggregate.
    output_path : str or pathlib.Path, optional
        Where to write the aggregated report. Defaults to a file in a new
        temporary directory, which is *not* cleaned up -- the returned ``Html``
        points at it, so the caller owns it from then on.

    Returns
    -------
    fileformats.text.unicode.Html
        The aggregated report, as a ``fileformats`` file object ready to be
        attached to a data row or handed to a pipeline.

    Raises
    ------
    FileNotFoundError
        If any of the given report paths does not exist.
    """
    paths = [Path(p) for p in report_paths]
    documents = load_reports(paths)
    aggregated = combine_reports(documents, sources=paths)

    if output_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="phi-finder-aggregate-")
        destination = Path(tmp_dir) / _DEFAULT_FILENAME
    else:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(aggregated, encoding="utf-8")

    return Html(destination)
