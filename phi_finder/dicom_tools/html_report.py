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

# What each provenance stamp is called in the report's "Profile" column, and
# the CSS class its pill gets. Records written before provenance was recorded
# carry no source and are left unlabelled.
_SOURCE_LABELS: "dict[str, tuple[str, str]]" = {
    ps3_15.SOURCE_PS3_15: ("PS3.15", "pill-ps315"),
    anonymise_dicom.SOURCE_NER: ("NER model", "pill-ner"),
}


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


def _display_tag(tag: pydicom.tag.Tag) -> str:
    """Formats a tag as ``(GGGG,EEEE)`` for display."""
    return f"({tag.group:04X},{tag.element:04X})"


def _tidy_tag(tag: str) -> str:
    """Formats a recorded tag string (``"(0010, 0010)"``) the same way.

    Parameters
    ----------
    tag : str
        A tag as written in the audit element, i.e. ``str(Tag)``.

    Returns
    -------
    str
        The tag as ``(GGGG,EEEE)``, or the string unchanged when it does not
        look like a tag (so nothing is lost from an unexpected record).
    """
    digits = re.sub(r"[^0-9a-fA-F]", "", tag or "")
    if len(digits) != 8:
        return tag
    return f"({digits[:4].upper()},{digits[4:].upper()})"


def _resolve_source(path: tuple, sources: dict, removed: bool) -> str:
    """Finds which engine de-identified the element at ``path``.

    The audit record names the element an action was applied to, which for a
    deleted sequence is the sequence itself rather than each value inside it,
    so nested values inherit their enclosing sequence's provenance. It is keyed
    by tag alone, so two elements sharing a tag in different places in the tree
    cannot be told apart; a nested value that is *gone* is therefore credited
    to the sequence that held it before its own tag is tried, since that is
    what usually took it away.

    Parameters
    ----------
    path : tuple
        An element path as produced by ``_walk_values``.
    sources : dict
        Maps a tag (in ``str(Tag)`` form, as recorded in the audit element) to
        the source that de-identified it.
    removed : bool
        Whether the element is gone from the dataset altogether.

    Returns
    -------
    str
        The source, or the empty string when neither the element nor any
        sequence holding it is in the audit record (e.g. a file anonymised
        before provenance was recorded).
    """
    tags = path[0::2]  # enclosing sequence tags, then the element's own
    order = list(reversed(tags))  # own tag first, then innermost sequence out
    if removed and len(tags) > 1:
        order = list(reversed(tags[:-1])) + [tags[-1]]
    for tag in order:
        source = sources.get(str(tag))
        if source:
            return source
    return ""


def collect_value_diffs(snapshot: dict, ds: pydicom.dataset.Dataset) -> list[dict]:
    """Diffs a pre-anonymisation snapshot against the de-identified dataset.

    Every element whose value changed is reported, not only the long free-text
    ones: short identifiers, dates and UIDs included. Each diff is stamped with
    the provenance phi-finder recorded for that element (see
    ``read_flagged_headers``), so the report can say what de-identified it.

    Parameters
    ----------
    snapshot : dict
        The pre-anonymisation snapshot from ``snapshot_values``.
    ds : pydicom.dataset.Dataset
        The same dataset after anonymisation.

    Returns
    -------
    list of dict
        One ``{"name", "tag", "location", "original", "redacted", "removed",
        "note", "private", "source"}`` entry per changed value.
        ``"location"`` is the element's full path
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
        otherwise bury the fields a reader is looking for. ``"source"`` is
        ``ps3_15.SOURCE_PS3_15`` or ``anonymise_dicom.SOURCE_NER``, or the
        empty string for a value phi-finder recorded no provenance for.
    """
    current = {path: value for path, _name, value in _walk_values(ds)}
    sources = {
        header.get("tag"): header.get("source", "")
        for header in read_flagged_headers(ds)
    }
    diffs = []
    for path, (name, original) in snapshot.items():
        removed = path not in current
        redacted = current.get(path, "")
        if redacted == original:
            continue
        diffs.append({
            "name": name,
            "tag": _display_tag(pydicom.tag.Tag(path[-1])),
            "location": _path_label(path, name),
            "original": original,
            "redacted": redacted,
            "removed": removed,
            "note": len(original) >= _CLINICAL_NOTE_MIN_LENGTH,
            "private": bool(pydicom.tag.Tag(path[-1]).is_private),
            "source": _resolve_source(path, sources, removed),
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


def _source_pill(source: str) -> str:
    """Renders a field's provenance as a pill, or a dash when unrecorded."""
    label, css_class = _SOURCE_LABELS.get(source, ("", ""))
    if not label:
        return '<span class="muted">&mdash;</span>'
    return f'<span class="pill {css_class}">{label}</span>'


def _render_value_rows(rows_by_field: dict) -> str:
    """Renders changed fields as before/after table rows.

    Fields are listed alphabetically; the values of a field keep the order they
    were met in, capped at ``_MAX_VALUES_PER_FIELD`` so a field that differs in
    every slice (a UID, say) cannot fill the report.

    Parameters
    ----------
    rows_by_field : dict
        Maps a field's location to a ``{"tag", "source", "values"}`` group, as
        gathered by ``build_html_report``; ``"values"`` holds its
        ``(original, redacted, removed)`` triples.

    Returns
    -------
    str
        The ``<tr>`` rows, or the empty string when there is nothing to show.
    """
    rows = []
    for field in sorted(rows_by_field, key=str.casefold):
        group = rows_by_field[field]
        values = group["values"]
        field_cell = html.escape(field) if field else "<em>(unnamed field)</em>"
        tag_cell = f'<td class="tag">{html.escape(group["tag"])}</td>'
        profile_cell = f'<td class="profile">{_source_pill(group["source"])}</td>'
        for original, redacted, removed in values[:_MAX_VALUES_PER_FIELD]:
            before = html.escape(original) if original else "<em>(blank)</em>"
            if removed:
                after = "<em>(field removed)</em>"
            else:
                after = html.escape(redacted) if redacted else "<em>(emptied)</em>"
            rows.append(
                f"      <tr><td>{field_cell}</td>{tag_cell}"
                f'<td class="before">{before}</td>'
                f'<td class="after">{after}</td>{profile_cell}</tr>'
            )
        extra = len(values) - _MAX_VALUES_PER_FIELD
        if extra > 0:
            # Say what was left out: a silently truncated table reads as if the
            # field only ever held the values shown.
            rows.append(
                f"      <tr><td>{field_cell}</td>{tag_cell}"
                f'<td class="more" colspan="3">&hellip; and {extra} further '
                "distinct value(s) for this field in this session, not shown."
                "</td></tr>"
            )
    return "\n".join(rows)


def _render_field_rows(flagged_headers: list[dict]) -> str:
    """Renders flagged headers as name/tag/profile rows, without their values.

    This is what a report built with no ``value_diffs`` shows: the same table,
    minus the two columns that would reproduce the PHI.

    Parameters
    ----------
    flagged_headers : list of dict
        Flagged-header records as produced by ``read_flagged_headers``.

    Returns
    -------
    str
        The ``<tr>`` rows, or the empty string when there is nothing to show.
    """
    by_key = {}
    for header in flagged_headers:
        name = (header.get("name") or "").strip()
        if not name:
            continue
        by_key[header.get("tag") or name] = header
    rows = []
    for header in sorted(by_key.values(), key=lambda h: h["name"].casefold()):
        tag = _tidy_tag(header.get("tag") or "")
        rows.append(
            f'      <tr><td>{html.escape(header["name"])}</td>'
            f'<td class="tag">{html.escape(tag)}</td>'
            f'<td class="profile">{_source_pill(header.get("source", ""))}</td></tr>'
        )
    return "\n".join(rows)


def _value_table(rows: str, with_values: bool = True) -> str:
    """Wraps rendered rows in a horizontally scrollable table."""
    value_headers = (
        "<th>Original value</th><th>After de-identification</th>"
        if with_values else ""
    )
    return (
        '    <div class="table-wrap">\n'
        "    <table>\n"
        "      <thead>\n"
        f"      <tr><th>Field</th><th>Tag</th>{value_headers}<th>Profile</th></tr>\n"
        "      </thead>\n"
        "      <tbody>\n"
        f"{rows}\n"
        "      </tbody>\n"
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
        are de-duplicated here). Only used when ``value_diffs`` is not given,
        as the fallback for a report that names the de-identified fields
        without reproducing their values. Each entry needs a ``"name"`` key;
        ``"tag"`` is used as the de-duplication key when present, and
        ``"source"`` (``ps3_15.SOURCE_PS3_15`` or
        ``anonymise_dicom.SOURCE_NER``) fills the Profile column. Entries with
        no ``"source"`` -- rebuilt from files anonymised before provenance was
        recorded -- show a dash there.
    n_images : int
        Number of images (DICOM files) processed in the session, shown in the
        summary.
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

    def esc(value: object) -> str:
        return html.escape(str(value))

    # De-duplicate identical diffs (the same field is redacted the same way in
    # every slice of a series), splitting them into the notes shown as
    # word-level diffs and the ones tabulated -- private fields apart.
    seen: set[tuple] = set()
    diff_blocks = []
    rows_by_field: dict[str, dict] = {}
    private_rows_by_field: dict[str, dict] = {}
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
            field = group.setdefault(location, {
                "tag": diff.get("tag", ""),
                "source": diff.get("source", ""),
                "values": [],
            })
            field["values"].append((original, redacted, removed))
            continue
        heading = esc(location) if location else "Clinical note"
        pill = _source_pill(diff.get("source", ""))
        if removed:
            # The element is gone, so there is nothing to diff against: show
            # the whole original as deleted and say so, rather than letting it
            # look like a thorough in-place redaction.
            badge = '<span class="badge badge-removed">removed entirely</span>'
            body = f'      <p class="diff diff-removed"><del>{esc(original)}</del></p>'
        else:
            badge = '<span class="badge badge-redacted">redacted in place</span>'
            body = f'      <p class="diff">{_render_text_diff(original, redacted)}</p>'
        diff_blocks.append(f"      <h3>{heading} {badge} {pill}</h3>\n{body}")

    if diff_blocks:
        diff_section = (
            "\n    <section>\n      <h2>Clinical notes</h2>\n"
            "      <p>The following free-text note(s) contained personal or "
            "health information. A field marked <em>redacted in place</em> is "
            "still in the image, with the removed text struck through and its "
            "replacement underlined; a field marked <em>removed entirely</em> "
            "was deleted from the image altogether, so its whole original "
            "value is struck through.</p>\n"
            '      <p class="warning">This section reproduces the original '
            "information and must be handled accordingly.</p>\n"
            + "\n".join(diff_blocks) + "\n    </section>"
        )
    else:
        diff_section = ""

    # The changed values, as a before/after table. The private (manufacturer)
    # fields go in a folded-away table of their own: a scanner writes hundreds
    # of them, and they would bury the standard fields.
    table_rows = _render_value_rows(rows_by_field)
    private_rows = _render_value_rows(private_rows_by_field)
    n_fields = len(rows_by_field) + len(private_rows_by_field) + len(diff_blocks)

    if table_rows or private_rows:
        notes_pointer = (
            " Longer free-text fields are shown in full under "
            "<em>Clinical notes</em> below." if diff_blocks else ""
        )
        table_section = (
            "\n    <section>\n      <h2>Changed fields</h2>\n"
            "      <p>Every field whose value changed is listed below, as it "
            "was before and after de-identification. <em>(field removed)</em> "
            "means the field was deleted from the image; <em>(emptied)</em> "
            "means it was kept but left blank." + notes_pointer + " The "
            "<strong>Profile</strong> column says what de-identified it: "
            "<em>PS3.15</em> is the fixed action the DICOM Basic Confidentiality "
            "Profile prescribes for that field, <em>NER model</em> means "
            "phi-finder's text-recognition models read the value and removed "
            "what looked like personal information.</p>\n"
            '      <p class="warning">This section reproduces the original '
            "values, including the personal information that was removed, and "
            "must be handled accordingly.</p>"
        )
        if table_rows:
            table_section += "\n" + _value_table(table_rows)
        if private_rows:
            n_private = len(private_rows_by_field)
            table_section += (
                "\n    <details>\n"
                f"      <summary>{n_private} private (manufacturer-defined) "
                "field(s) also changed</summary>\n"
                "      <p>Private fields are written by the scanner "
                "manufacturer rather than defined by the DICOM standard.</p>\n"
                + _value_table(private_rows) + "\n    </details>"
            )
        table_section += "\n    </section>"
    elif flagged_headers:
        # No values to show (the caller passed no diffs), so report the same
        # table without the two columns that would reproduce the PHI.
        field_rows = _render_field_rows(flagged_headers)
        n_fields = field_rows.count("<tr>")
        table_section = (
            "\n    <section>\n      <h2>Changed fields</h2>\n"
            "      <p>phi-finder de-identified the header fields listed below. "
            "The <strong>Profile</strong> column says what de-identified each "
            "one: <em>PS3.15</em> is the fixed action the DICOM Basic "
            "Confidentiality Profile prescribes for that field, <em>NER "
            "model</em> means phi-finder's text-recognition models read the "
            "value and removed what looked like personal information.</p>\n"
            + _value_table(field_rows, with_values=False) + "\n    </section>"
        )
    else:
        table_section = (
            "\n    <section>\n      <p>phi-finder found no personal or health "
            "information to remove from the image header fields.</p>\n"
            "    </section>"
        )

    stats = [(esc(n_images), "image(s) processed"), (esc(n_fields), "field(s) changed")]
    if session_id:
        stats.append((esc(session_id), "session"))
    if use_case:
        stats.append((esc(use_case), "method"))
    summary = "\n".join(
        f'      <div class="stat"><span class="stat-value">{value}</span>'
        f'<span class="stat-label">{label}</span></div>'
        for value, label in stats
    )

    footer = "Generated by phi-finder on " + esc(
        generated_at.strftime("%d %B %Y at %H:%M")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>De-identification report</title>
  <style>
    :root {{ --bg: #f4f6f8; --card: #fff; --ink: #1c2530; --muted: #67717e;
             --line: #e3e7ec; --accent: #2f5d8c; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica,
            Arial, sans-serif; background: var(--bg); color: var(--ink);
            margin: 0; padding: 2rem 1rem; line-height: 1.55; }}
    main {{ max-width: 1040px; margin: 0 auto; background: var(--card);
            border: 1px solid var(--line); border-radius: 10px;
            padding: 1.75rem 2rem 2rem;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06); }}
    h1 {{ font-size: 1.6rem; margin: 0 0 0.2rem; letter-spacing: -0.01em; }}
    .subtitle {{ margin: 0 0 1.25rem; color: var(--muted); }}
    h2 {{ font-size: 1.15rem; margin: 0 0 0.5rem; padding-bottom: 0.3rem;
          border-bottom: 2px solid var(--accent); display: inline-block; }}
    h3 {{ font-size: 0.95rem; margin: 1.25rem 0 0.4rem; }}
    section {{ margin-top: 2rem; }}
    p {{ margin: 0.5rem 0 0.9rem; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
    .stat {{ flex: 1 1 8rem; background: #f8fafc; border: 1px solid var(--line);
             border-radius: 8px; padding: 0.55rem 0.8rem; }}
    .stat-value {{ display: block; font-size: 1.2rem; font-weight: 600;
                   overflow-wrap: anywhere; }}
    .stat-label {{ display: block; font-size: 0.7rem; color: var(--muted);
                   text-transform: uppercase; letter-spacing: 0.05em; }}
    .warning {{ background: #fff8e6; border: 1px solid #f0dca6;
                border-left: 4px solid #d9a406; border-radius: 6px;
                padding: 0.6rem 0.9rem; font-size: 0.9rem; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line);
                   border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
    th, td {{ text-align: left; vertical-align: top; padding: 0.45rem 0.6rem;
              border-bottom: 1px solid var(--line); overflow-wrap: anywhere; }}
    thead th {{ position: sticky; top: 0; background: #f8fafc; color: var(--muted);
                font-size: 0.72rem; text-transform: uppercase;
                letter-spacing: 0.05em; white-space: nowrap; }}
    tbody tr:last-child td {{ border-bottom: none; }}
    td.tag, td.before, td.after {{ font-family: ui-monospace, SFMono-Regular,
                                   Menlo, Consolas, monospace; font-size: 0.8rem; }}
    td.tag {{ color: var(--muted); white-space: nowrap; }}
    td.before {{ background: #fff5f5; color: #9b2226; }}
    td.after {{ background: #f2faf5; color: #14663a; }}
    td.profile {{ white-space: nowrap; }}
    td.more {{ color: var(--muted); font-style: italic; }}
    td em {{ color: var(--muted); }}
    .pill, .badge {{ display: inline-block; font-size: 0.7rem; font-weight: 600;
                     letter-spacing: 0.02em; padding: 0.05rem 0.45rem;
                     border-radius: 999px; white-space: nowrap;
                     vertical-align: middle; }}
    .pill-ps315 {{ background: #e9f0fa; color: #24547f; border: 1px solid #cbdcf2; }}
    .pill-ner {{ background: #eeeffd; color: #443c9c; border: 1px solid #d5d7f6; }}
    .badge-redacted {{ background: #eaf6ee; color: #14663a; border: 1px solid #c7e6d2; }}
    .badge-removed {{ background: #fdecec; color: #9b2226; border: 1px solid #f5cccc; }}
    .muted {{ color: var(--muted); }}
    .diff {{ white-space: pre-wrap; background: #fbfcfd; border: 1px solid var(--line);
             border-radius: 8px; padding: 0.8rem 1rem; max-width: 80ch;
             font-size: 0.92rem; }}
    .diff-removed {{ background: #fffafa; border-color: #f2d5d5; }}
    del {{ background: #ffdcdc; color: #8a1c22; }}
    ins {{ background: #d6f5e0; color: #14663a; text-decoration: none; }}
    details {{ margin-top: 1rem; background: #fbfcfd; border: 1px solid var(--line);
               border-radius: 8px; padding: 0.6rem 0.8rem; }}
    summary {{ cursor: pointer; font-weight: 600; color: var(--accent); }}
    details p {{ color: var(--muted); font-size: 0.85rem; }}
    footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--line);
              font-size: 0.8rem; color: var(--muted); }}
    @media print {{
      body {{ background: #fff; padding: 0; }}
      main {{ border: none; box-shadow: none; max-width: none; padding: 0; }}
      thead th {{ position: static; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>De-identification report</h1>
    <p class="subtitle">What phi-finder removed from this session</p>
    <div class="stats">
{summary}
    </div>
{table_section}
{diff_section}
    <footer>{footer}</footer>
  </main>
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
