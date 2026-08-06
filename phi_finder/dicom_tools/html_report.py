import difflib
import html
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pydicom
from pydicom.datadict import dictionary_description
from fileformats.generic import File
from frametree.core.entry import DataEntry
from frametree.core.row import DataRow


# Free-text VRs whose value can hold a clinical note (e.g. radiology report).
_NOTE_TEXT_VRS = frozenset({"LO", "LT", "SH", "ST", "UC", "UT"})
# Minimum original length (characters) for text to be shown as a before/after diff. 
_CLINICAL_NOTE_MIN_LENGTH = 60
# phi-finder's own audit element written by anonymise_image; never a note.
_AUDIT_TAG = 0x02091000


def _walk_note_text(ds: pydicom.dataset.Dataset,
                    prefix: tuple = ()) -> "list[tuple]":
    """Yields ``(path, name, value)`` for each free-text element in a dataset.

    Recurses into sequences so notes nested inside them are reached. ``path``
    is a hashable tuple identifying the element's position (its tag, plus the
    tag/index pairs of any enclosing sequence items).

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The dataset to walk.
    prefix : tuple
        Path components accumulated from enclosing sequences (used internally
        during recursion; callers pass the default).

    Yields
    ------
    tuple
        ``(path, name, value)`` where ``path`` is a tuple, ``name`` is the
        element's human-readable name and ``value`` is its text (multi-valued
        elements are joined with a single space).
    """
    for elem in ds:
        if elem.tag == _AUDIT_TAG or elem.tag.is_private_creator:
            continue
        if elem.VR == "SQ":
            for i, sub_ds in enumerate(elem.value):
                if isinstance(sub_ds, pydicom.dataset.Dataset):
                    yield from _walk_note_text(sub_ds, prefix + (elem.tag, i))
            continue
        if elem.VR not in _NOTE_TEXT_VRS:
            continue
        value = elem.value
        if value is None:
            continue
        if isinstance(value, pydicom.multival.MultiValue):
            value = " ".join(str(v) for v in value)
        else:
            value = str(value)
        yield prefix + (elem.tag,), elem.name, value


def _tag_name(tag: pydicom.tag.Tag) -> str:
    """Returns a tag's human-readable name, falling back to its numeric form.

    Parameters
    ----------
    tag : pydicom.tag.Tag
        The tag to describe.

    Returns
    -------
    str
        The DICOM dictionary description, or ``str(tag)`` for private and
        unknown tags that have no entry.
    """
    try:
        return dictionary_description(tag)
    except (KeyError, ValueError):
        return str(tag)


def _path_label(path: tuple, leaf_name: str = "") -> str:
    """Renders an element ``path`` as a readable location.

    A note nested inside sequences is reported as, e.g., ``"Content Sequence
    [4] > Content Sequence [0] > Text Value"``, so two same-named elements in
    different places in the tree can be told apart.

    Parameters
    ----------
    path : tuple
        An element path as produced by ``_walk_note_text``: alternating
        sequence tag and item index, ending with the element's own tag.
    leaf_name : str, optional
        Name to use for the final component. Defaults to looking the tag up.

    Returns
    -------
    str
        The formatted location.
    """
    parts = [
        f"{_tag_name(path[i])} [{path[i + 1]}]" for i in range(0, len(path) - 1, 2)
    ]
    parts.append(leaf_name or _tag_name(path[-1]))
    return " > ".join(parts)


def snapshot_long_text(ds: pydicom.dataset.Dataset) -> dict:
    """Records the free-text fields of a dataset before it is anonymised.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The dataset before anonymisation mutates it in place.

    Returns
    -------
    dict
        Maps each element ``path`` (see ``_walk_note_text``) to a
        ``(name, original_value)`` tuple.
    """
    return {path: (name, value) for path, name, value in _walk_note_text(ds)}


def collect_note_diffs(snapshot: dict, ds: pydicom.dataset.Dataset) -> list[dict]:
    """Diffs a pre-anonymisation text snapshot against the redacted dataset.

    Only fields at least ``_CLINICAL_NOTE_MIN_LENGTH`` characters long and
    whose value actually changed are reported, so short identifiers (already
    summarised by name in the header list) are not treated as clinical notes.

    Parameters
    ----------
    snapshot : dict
        The pre-anonymisation snapshot from ``snapshot_long_text``.
    ds : pydicom.dataset.Dataset
        The same dataset after anonymisation.

    Returns
    -------
    list of dict
        One ``{"name", "location", "original", "redacted", "removed"}`` entry
        per changed note. ``"location"`` is the element's full path through any
        enclosing sequences, which distinguishes same-named notes in different
        places in the tree. ``"removed"`` is True when the element is gone from
        the dataset altogether (e.g. the sequence holding it was emptied by the
        PS3.15 profile) rather than redacted in place — both leave no text
        behind, but only the latter means the value itself was scanned. The
        ``"original"`` value contains the un-redacted PHI, so callers must
        treat the result as sensitive.
    """
    current = {path: value for path, _name, value in _walk_note_text(ds)}
    diffs = []
    for path, (name, original) in snapshot.items():
        if len(original) < _CLINICAL_NOTE_MIN_LENGTH:
            continue
        removed = path not in current
        redacted = current.get(path, "")
        if redacted == original:
            continue
        diffs.append({
            "name": name,
            "location": _path_label(path, name),
            "original": original,
            "redacted": redacted,
            "removed": removed,
        })
    return diffs


def _render_text_diff(original: str, redacted: str) -> str:
    """Renders an inline, word-level HTML diff of ``original`` vs ``redacted``.

    Removed segments (the PHI that was taken out) are wrapped in ``<del>`` and
    inserted segments (the placeholder that replaced it) in ``<ins>``. Both
    sides are HTML-escaped, and whitespace tokens are preserved so the note's
    original line breaks survive (the ``.diff`` style uses ``white-space:
    pre-wrap``).

    Parameters
    ----------
    original : str
        The text before redaction.
    redacted : str
        The text after redaction.

    Returns
    -------
    str
        An HTML fragment representing the diff.
    """
    # Split into word and whitespace tokens so the diff is word-level yet
    # reconstructs the exact original text (whitespace included).
    orig_tokens = re.findall(r"\S+|\s+", original)
    redr_tokens = re.findall(r"\S+|\s+", redacted)
    matcher = difflib.SequenceMatcher(a=orig_tokens, b=redr_tokens, autojunk=False)
    parts = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        old = html.escape("".join(orig_tokens[i1:i2]))
        new = html.escape("".join(redr_tokens[j1:j2]))
        if op == "equal":
            parts.append(old)
        elif op == "delete":
            parts.append(f"<del>{old}</del>")
        elif op == "insert":
            parts.append(f"<ins>{new}</ins>")
        else:  # replace
            parts.append(f"<del>{old}</del><ins>{new}</ins>")
    return "".join(parts)


def read_flagged_headers(ds: pydicom.dataset.Dataset) -> list[dict]:
    """Reads the list of PHI-flagged headers phi-finder recorded in a dataset.

    ``anonymise_dicom.anonymise_image`` writes a private audit element at
    ``(0209,1000)`` (VR ``UT``, creator ``"phi-finder"``) holding a JSON list
    of ``{"tag", "name"}`` dicts, one per header whose value was scrubbed.
    This reads and parses that element.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        A DICOM dataset previously anonymised by phi-finder.

    Returns
    -------
    list of dict
        One ``{"tag": str, "name": str}`` entry per flagged header. Empty when
        the audit tag is absent or unreadable, so callers building a report
        never crash on an un-anonymised or malformed file.
    """
    try:
        block = ds.private_block(0x0209, "phi-finder")
        headers = json.loads(block[0x00].value)
    except (KeyError, ValueError, TypeError):
        return []
    return headers if isinstance(headers, list) else []


def build_html_report(flagged_headers: list[dict],
                      n_images: int,
                      session_id: str | None = None,
                      use_case: str | None = None,
                      generated_at: datetime | None = None,
                      note_diffs: list[dict] | None = None) -> str:
    """Builds a plain-language HTML de-identification report for one session.

    Parameters
    ----------
    flagged_headers : list of dict
        The flagged headers accumulated over the session, as produced by
        ``read_flagged_headers`` (may contain duplicates across images; they
        are de-duplicated here). Each entry needs a ``"name"`` key; ``"tag"``
        is used as the de-duplication key when present.
    n_images : int
        Number of images (DICOM files) processed in the session, shown in the
        summary line.
    session_id : str, optional
        Human-readable session/study identifier, shown in the summary when given.
    use_case : str, optional
        The de-identification use case applied (e.g. ``'Standard'``,
        ``'PS3.15'``), shown in the summary when given.
    generated_at : datetime.datetime, optional
        Timestamp shown in the footer. Defaults to ``datetime.now()``; accept a
        value to make the output deterministic (e.g. in tests).
    note_diffs : list of dict, optional
        Clinical-note diffs accumulated over the session, as produced by
        ``collect_note_diffs``. Each entry needs ``"name"``, ``"original"`` and
        ``"redacted"`` keys, and may carry ``"location"`` (used in place of the
        name as the heading) and ``"removed"`` (rendered as a deleted rather
        than a redacted field). Identical diffs (same note repeated across
        slices) are de-duplicated. Defaults to no diffs.

    Returns
    -------
    str
        A self-contained HTML document (inline styles, no external assets).
    """
    if generated_at is None:
        generated_at = datetime.now()

    # De-duplicate by tag when available (stable identity), else by name.
    unique: dict[str, str] = {}
    for header in flagged_headers:
        name = (header.get("name") or "").strip()
        if not name:
            continue
        key = header.get("tag") or name
        unique[key] = name
    names = sorted(set(unique.values()), key=str.casefold)

    def esc(value: object) -> str:
        return html.escape(str(value))

    summary_bits = [f"<strong>{n_images}</strong> image(s) processed"]
    if session_id:
        summary_bits.append(f"session <strong>{esc(session_id)}</strong>")
    if use_case:
        summary_bits.append(f"method <strong>{esc(use_case)}</strong>")
    summary = " &middot; ".join(summary_bits)

    if names:
        intro = (
            "phi-finder found and removed the following types of personal or "
            "health information from the image header fields:"
        )
        items = "\n".join(f"      <li>{esc(name)}</li>" for name in names)
        findings = f"    <p>{intro}</p>\n    <ul>\n{items}\n    </ul>"
    else:
        findings = (
            "    <p>phi-finder found no personal or health information to "
            "remove from the image header fields.</p>"
        )

    # De-duplicate identical note diffs (the same report repeats across slices).
    seen: set[tuple[str, str, str, bool]] = set()
    diff_blocks = []
    for diff in note_diffs or []:
        original = diff.get("original", "")
        redacted = diff.get("redacted", "")
        name = (diff.get("name") or "").strip()
        location = (diff.get("location") or "").strip() or name
        removed = bool(diff.get("removed"))
        key = (location, original, redacted, removed)
        if key in seen:
            continue
        seen.add(key)
        heading = esc(location) if location else "Clinical note"
        if removed:
            # The element is gone, so there is nothing to diff against: show
            # the whole original as deleted and say so, rather than letting it
            # look like a thorough in-place redaction.
            badge = '<span class="badge badge-removed">removed entirely</span>'
            body = f'    <p class="diff diff-removed"><del>{esc(original)}</del></p>'
        else:
            badge = '<span class="badge badge-redacted">redacted in place</span>'
            body = f'    <p class="diff">{_render_text_diff(original, redacted)}</p>'
        diff_blocks.append(f"    <h3>{heading} {badge}</h3>\n{body}")

    if diff_blocks:
        diff_intro = (
            "    <p>The following free-text note(s) contained personal or "
            "health information. A field marked <em>redacted in place</em> is "
            "still in the image, with the removed text struck through and its "
            "replacement underlined; a field marked <em>removed entirely</em> "
            "was deleted from the image altogether, so its whole original "
            "value is struck through. <strong>This section reproduces the "
            "original information and must be handled accordingly.</strong></p>"
        )
        diff_section = "\n  <h2>Clinical notes</h2>\n" + diff_intro + "\n" + "\n".join(diff_blocks)
    else:
        diff_section = ""

    footer = "Generated by phi-finder on " + esc(
        generated_at.strftime("%d %B %Y at %H:%M")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>De-identification report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; color: #222;
           max-width: 720px; margin: 2rem auto; padding: 0 1rem;
           line-height: 1.5; }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1.2rem; margin-top: 2rem; }}
    h3 {{ font-size: 1rem; margin-bottom: 0.3rem; }}
    .summary {{ background: #f2f6fb; border: 1px solid #d7e2f0;
                border-radius: 6px; padding: 0.75rem 1rem; }}
    ul {{ padding-left: 1.4rem; }}
    li {{ margin: 0.15rem 0; }}
    .diff {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #eee;
             border-radius: 6px; padding: 0.75rem 1rem; }}
    .diff-removed {{ background: #fff7f7; border-color: #f0cdcd; }}
    .badge {{ font-size: 0.7rem; font-weight: normal; text-transform: uppercase;
              letter-spacing: 0.03em; padding: 0.1rem 0.4rem; border-radius: 3px;
              vertical-align: middle; white-space: nowrap; }}
    .badge-redacted {{ background: #e6f0e6; color: #060; border: 1px solid #cde0cd; }}
    .badge-removed {{ background: #fde8e8; color: #900; border: 1px solid #f0cdcd; }}
    del {{ background: #fdd; color: #900; }}
    ins {{ background: #dfd; color: #060; text-decoration: none; }}
    footer {{ margin-top: 2rem; font-size: 0.8rem; color: #777; }}
  </style>
</head>
<body>
  <h1>De-identification report</h1>
  <p class="summary">{summary}</p>
{findings}
{diff_section}
  <footer>{footer}</footer>
</body>
</html>
"""


def save_html_report(data_row: DataRow,
                     html_report: str,
                     entry_name: str = "deidentification_report@deidentified") -> DataEntry:
    """Uploads an HTML de-identification report to a data row as a new entry.

    Existing entry is overwritten, so re-running the pipeline does not duplicate reports.

    Parameters
    ----------
    data_row : DataRow
        The data row (session) to attach the report to.
    html_report : str
        The HTML document to upload, e.g. as produced by ``build_html_report``.
    entry_name : str, optional
        The resource path of the report entry within the row. Defaults to
        ``'deidentification_report@deidentified'``.

    Returns
    -------
    DataEntry
        The created (or re-used) entry holding the uploaded report.
    """
    with tempfile.TemporaryDirectory(prefix="phi-finder-report-") as tmp_dir:
        report_path = Path(tmp_dir) / "deidentification_report.html"
        report_path.write_text(html_report, encoding="utf-8")

        entries_names = [
            k[0] if isinstance(k, tuple) else k
            for k in data_row.entries_dict.keys()
        ]
        if entry_name in entries_names:
            entry = data_row.entry(entry_name)
        else:
            entry = data_row.create_entry(entry_name, datatype=File)

        # Assignment uploads the file while the temp dir is still alive.
        entry.item = File(report_path)
    return entry
