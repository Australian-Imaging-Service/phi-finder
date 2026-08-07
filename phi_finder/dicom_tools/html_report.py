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

from phi_finder.dicom_tools import anonymise_dicom, ps3_15


# VRs holding bulk binary data (pixels, overlays, lookup tables, unparsed
# private blobs). Their values are not human-readable and can be enormous, so
# they are never snapshotted or diffed.
_BINARY_VRS = frozenset({
    "OB", "OW", "OF", "OD", "OL", "OV", "OB or OW", "OW or OB", "UN",
})
# Longest multi-valued element still reported. Anything longer is bulk numeric
# data (e.g. a lookup table), not a value a reader would recognise as PHI.
_MAX_MULTIVALUE_ITEMS = 64
# Minimum original length (characters) for text to be shown as a before/after diff.
_CLINICAL_NOTE_MIN_LENGTH = 60
# Distinct values per field listed in the changed-values table. A field whose
# value differs per slice (e.g. a UID) would otherwise fill the report.
_MAX_VALUES_PER_FIELD = 5
# phi-finder's own audit element written by anonymise_image; never a note.
_AUDIT_TAG = 0x02091000

# Header findings are grouped by the "source" each record carries, in this
# order. Records written before provenance was recorded have no source; they
# fall into the trailing unlabelled group, which keeps their old rendering.
_FINDING_SECTIONS: "list[tuple[str, str, str]]" = [
    (
        ps3_15.SOURCE_PS3_15,
        "Removed by the DICOM PS3.15 profile",
        "These header fields were de-identified by the DICOM PS3.15 Annex E "
        "Basic Application Level Confidentiality Profile, which prescribes a "
        "fixed action for each of them regardless of what they contained:",
    ),
    (
        anonymise_dicom.SOURCE_NER,
        "Found by the text-scanning models",
        "The value of these header fields was read by phi-finder's "
        "text-recognition models, which removed the parts that looked like "
        "personal or health information:",
    ),
    (
        "",
        "",
        "phi-finder found and removed the following types of personal or "
        "health information from the image header fields:",
    ),
]


def _walk_values(ds: pydicom.dataset.Dataset,
                 prefix: tuple = ()) -> "list[tuple]":
    """Yields ``(path, name, value)`` for each readable element in a dataset.

    Every element is reported, whatever its VR, except the bulk binary ones
    (``_BINARY_VRS``, which cover pixel data, overlays and unparsed private
    blobs) and multi-valued elements longer than ``_MAX_MULTIVALUE_ITEMS``.
    Recurses into sequences so values nested inside them are reached. ``path``
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
        elements are joined with a single space, and an absent value is the
        empty string, so an emptied element is told apart from a deleted one).
    """
    for elem in ds:
        if elem.tag == _AUDIT_TAG or elem.tag.is_private_creator:
            continue
        if elem.VR == "SQ":
            for i, sub_ds in enumerate(elem.value):
                if isinstance(sub_ds, pydicom.dataset.Dataset):
                    yield from _walk_values(sub_ds, prefix + (elem.tag, i))
            continue
        if elem.VR in _BINARY_VRS:
            continue
        value = elem.value
        if value is None:
            value = ""
        elif isinstance(value, pydicom.multival.MultiValue):
            if len(value) > _MAX_MULTIVALUE_ITEMS:
                continue
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


def snapshot_values(ds: pydicom.dataset.Dataset) -> dict:
    """Records the header values of a dataset before it is anonymised.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The dataset before anonymisation mutates it in place.

    Returns
    -------
    dict
        Maps each element ``path`` (see ``_walk_values``) to a
        ``(name, original_value)`` tuple.
    """
    return {path: (name, value) for path, name, value in _walk_values(ds)}


def collect_value_diffs(snapshot: dict, ds: pydicom.dataset.Dataset) -> list[dict]:
    """Diffs a pre-anonymisation snapshot against the de-identified dataset.

    Every element whose value changed is reported, not only the long free-text
    ones: short identifiers, dates and UIDs included.

    Parameters
    ----------
    snapshot : dict
        The pre-anonymisation snapshot from ``snapshot_values``.
    ds : pydicom.dataset.Dataset
        The same dataset after anonymisation.

    Returns
    -------
    list of dict
        One ``{"name", "location", "original", "redacted", "removed", "note"}``
        entry per changed value. ``"location"`` is the element's full path
        through any enclosing sequences, which distinguishes same-named fields
        in different places in the tree. ``"removed"`` is True when the element
        is gone from the dataset altogether (e.g. deleted by the PS3.15 profile,
        or the sequence holding it was emptied) rather than rewritten in place —
        both can leave no text behind, but only the latter means the value
        itself was scanned. ``"note"`` is True for values at least
        ``_CLINICAL_NOTE_MIN_LENGTH`` characters long, which ``build_html_report``
        renders as word-level diffs rather than as table rows. The
        ``"original"`` value contains the un-redacted PHI, so callers must
        treat the result as sensitive. ``"private"`` marks a private
        (manufacturer-defined) attribute, which the report tabulates apart from
        the standard ones: a scanner writes hundreds of them and they would
        otherwise bury the fields a reader is looking for.
    """
    current = {path: value for path, _name, value in _walk_values(ds)}
    diffs = []
    for path, (name, original) in snapshot.items():
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
            "note": len(original) >= _CLINICAL_NOTE_MIN_LENGTH,
            "private": bool(pydicom.tag.Tag(path[-1]).is_private),
        })
    return diffs


def diff_key(diff: dict) -> tuple:
    """Returns the identity of a value diff, for de-duplication.

    The same field is redacted the same way in every slice of a series, so
    diffs repeat across images; this key is what makes those repeats one entry.

    Parameters
    ----------
    diff : dict
        A diff as produced by ``collect_value_diffs``.

    Returns
    -------
    tuple
        ``(location, original, redacted, removed)``, falling back to the
        element name when no location is recorded.
    """
    location = (diff.get("location") or diff.get("name") or "").strip()
    return (
        location,
        diff.get("original", ""),
        diff.get("redacted", ""),
        bool(diff.get("removed")),
    )


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


def _render_value_rows(rows_by_field: "dict[str, list[tuple]]") -> str:
    """Renders changed values as before/after table rows.

    Fields are listed alphabetically; the values of a field keep the order they
    were met in, capped at ``_MAX_VALUES_PER_FIELD`` so a field that differs in
    every slice (a UID, say) cannot fill the report.

    Parameters
    ----------
    rows_by_field : dict
        Maps a field's location to its ``(original, redacted, removed)``
        values, as gathered by ``build_html_report``.

    Returns
    -------
    str
        The ``<tr>`` rows, or the empty string when there is nothing to show.
    """
    rows = []
    for field in sorted(rows_by_field, key=str.casefold):
        values = rows_by_field[field]
        field_cell = html.escape(field) if field else "<em>(unnamed field)</em>"
        for original, redacted, removed in values[:_MAX_VALUES_PER_FIELD]:
            before = html.escape(original) if original else "<em>(blank)</em>"
            if removed:
                after = "<em>(field removed)</em>"
            else:
                after = html.escape(redacted) if redacted else "<em>(emptied)</em>"
            rows.append(
                f"      <tr><td>{field_cell}</td>"
                f'<td class="before">{before}</td>'
                f'<td class="after">{after}</td></tr>'
            )
        extra = len(values) - _MAX_VALUES_PER_FIELD
        if extra > 0:
            # Say what was left out: a silently truncated table reads as if the
            # field only ever held the values shown.
            rows.append(
                f"      <tr><td>{field_cell}</td>"
                f'<td class="more" colspan="2"><em>&hellip; and {extra} further '
                "distinct value(s) for this field in this session, not shown."
                "</em></td></tr>"
            )
    return "\n".join(rows)


def _value_table(rows: str) -> str:
    """Wraps rendered rows in a horizontally scrollable before/after table."""
    return (
        '    <div class="table-wrap">\n'
        "    <table>\n"
        "      <tr><th>Field</th><th>Original value</th>"
        "<th>After de-identification</th></tr>\n"
        f"{rows}\n"
        "    </table>\n"
        "    </div>"
    )


def read_flagged_headers(ds: pydicom.dataset.Dataset) -> list[dict]:
    """Reads the list of PHI-flagged headers phi-finder recorded in a dataset.

    ``anonymise_dicom.anonymise_image`` writes a private audit element at
    ``(0209,1000)`` (VR ``UT``, creator ``"phi-finder"``) holding a JSON list
    of ``{"tag", "name", "source"}`` dicts, one per header whose value was
    scrubbed, where ``"source"`` records what de-identified it. This reads and
    parses that element.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        A DICOM dataset previously anonymised by phi-finder.

    Returns
    -------
    list of dict
        One ``{"tag": str, "name": str, "source": str}`` entry per flagged
        header (``"source"`` absent in files anonymised by older
        versions). Empty when
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
                      value_diffs: list[dict] | None = None) -> str:
    """Builds a plain-language HTML de-identification report for one session.

    Parameters
    ----------
    flagged_headers : list of dict
        The flagged headers accumulated over the session, as produced by
        ``read_flagged_headers`` (may contain duplicates across images; they
        are de-duplicated here). Each entry needs a ``"name"`` key; ``"tag"``
        is used as the de-duplication key when present, and ``"source"``
        (``ps3_15.SOURCE_PS3_15`` or ``anonymise_dicom.SOURCE_NER``) selects
        which findings section it is listed under. Entries with no ``"source"``
        -- reports rebuilt from files anonymised before provenance was
        recorded -- are listed together in a single unlabelled section.
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
    value_diffs : list of dict, optional
        Before/after value diffs accumulated over the session, as produced by
        ``collect_value_diffs``. Each entry needs ``"name"``, ``"original"``
        and ``"redacted"`` keys, and may carry ``"location"`` (used in place of
        the name as the heading), ``"removed"`` (rendered as a deleted rather
        than a rewritten field) and ``"note"``. Entries with ``"note"`` False
        are tabulated as before/after rows under "Changed field values"; the
        rest (long free text, and any entry with no ``"note"`` key) are
        rendered as word-level diffs under "Clinical notes". Entries marked
        ``"private"`` are tabulated in a collapsed block of their own, so the
        hundreds a scanner writes do not bury the standard fields. Identical
        diffs (the same field redacted the same way in every slice) are
        de-duplicated, and at most ``_MAX_VALUES_PER_FIELD`` distinct values
        are tabulated per field. Defaults to no diffs.

    Returns
    -------
    str
        A self-contained HTML document (inline styles, no external assets).
    """
    if generated_at is None:
        generated_at = datetime.now()

    # Group by what de-identified the header, de-duplicating by tag when
    # available (stable identity) and else by name, within each group.
    grouped: dict[str, dict[str, str]] = {}
    for header in flagged_headers:
        name = (header.get("name") or "").strip()
        if not name:
            continue
        key = header.get("tag") or name
        grouped.setdefault(header.get("source") or "", {})[key] = name

    def esc(value: object) -> str:
        return html.escape(str(value))

    summary_bits = [f"<strong>{n_images}</strong> image(s) processed"]
    if session_id:
        summary_bits.append(f"session <strong>{esc(session_id)}</strong>")
    if use_case:
        summary_bits.append(f"method <strong>{esc(use_case)}</strong>")
    summary = " &middot; ".join(summary_bits)

    finding_blocks = []
    for source, title, intro in _FINDING_SECTIONS:
        names = sorted(set(grouped.get(source, {}).values()), key=str.casefold)
        if not names:
            continue
        items = "\n".join(f"      <li>{esc(name)}</li>" for name in names)
        heading = f"  <h2>{esc(title)}</h2>\n" if title else ""
        finding_blocks.append(f"{heading}    <p>{intro}</p>\n    <ul>\n{items}\n    </ul>")

    if finding_blocks:
        findings = "\n".join(finding_blocks)
    else:
        findings = (
            "    <p>phi-finder found no personal or health information to "
            "remove from the image header fields.</p>"
        )

    # De-duplicate identical diffs (the same field is redacted the same way in
    # every slice of a series).
    seen: set[tuple] = set()
    diff_blocks = []
    rows_by_field: dict[str, list[tuple[str, str, bool]]] = {}
    private_rows_by_field: dict[str, list[tuple[str, str, bool]]] = {}
    for diff in value_diffs or []:
        key = diff_key(diff)
        if key in seen:
            continue
        seen.add(key)
        location, original, redacted, removed = key
        # Entries carrying no "note" key were built by a caller by hand rather
        # than by collect_value_diffs; keep the long-form rendering they were
        # written for.
        if not diff.get("note", True):
            group = private_rows_by_field if diff.get("private") else rows_by_field
            group.setdefault(location, []).append((original, redacted, removed))
            continue
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

    # Every other changed value, as a before/after table. The private
    # (manufacturer) fields go in a folded-away table of their own: a scanner
    # writes hundreds of them, and they would bury the standard fields.
    table_rows = _render_value_rows(rows_by_field)
    private_rows = _render_value_rows(private_rows_by_field)

    if table_rows or private_rows:
        notes_pointer = (
            " Longer free-text fields are shown in full under "
            "<em>Clinical notes</em> below." if diff_blocks else ""
        )
        table_intro = (
            "    <p>Every field whose value changed is listed below, as it was "
            "before and after de-identification. <em>(field removed)</em> means "
            "the field was deleted from the image; <em>(emptied)</em> means it "
            "was kept but left blank." + notes_pointer + " <strong>This section "
            "reproduces the original values, including the personal information "
            "that was removed, and must be handled accordingly.</strong></p>"
        )
        table_section = "\n  <h2>Changed field values</h2>\n" + table_intro
        if table_rows:
            table_section += "\n" + _value_table(table_rows)
        if private_rows:
            n_private = len(private_rows_by_field)
            table_section += (
                "\n    <details>\n"
                f"      <summary>{n_private} private (manufacturer-defined) "
                "field(s) also changed &mdash; click to show</summary>\n"
                "      <p>Private fields are written by the scanner "
                "manufacturer rather than defined by the DICOM standard.</p>\n"
                + _value_table(private_rows) + "\n    </details>"
            )
    else:
        table_section = ""

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
           max-width: 860px; margin: 2rem auto; padding: 0 1rem;
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
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #e3e3e3; padding: 0.3rem 0.5rem;
              text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
    th {{ background: #f2f6fb; font-weight: bold; }}
    td.before {{ background: #fdf3f3; color: #900; }}
    td.after {{ background: #f3faf3; color: #060; }}
    td em, td.more {{ color: #777; }}
    details {{ margin-top: 0.75rem; }}
    summary {{ cursor: pointer; color: #345; }}
    footer {{ margin-top: 2rem; font-size: 0.8rem; color: #777; }}
  </style>
</head>
<body>
  <h1>De-identification report</h1>
  <p class="summary">{summary}</p>
{findings}
{table_section}
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
