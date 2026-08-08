import pytest
from fileformats.text.unicode import Html

import aggregate_reports


def _write_report(directory, name, body):
    path = directory / name
    path.write_text(
        f"<!DOCTYPE html><html><body><p>{body}</p></body></html>", encoding="utf-8"
    )
    return path


def test_aggregate_reports_returns_html_fileformat(tmp_path):
    paths = [
        _write_report(tmp_path, "one.html", "session one"),
        _write_report(tmp_path, "two.html", "session two"),
    ]

    result = aggregate_reports.aggregate_reports(paths)

    assert isinstance(result, Html)
    assert result.fspath.exists()
    assert result.fspath.suffix == ".html"


def test_aggregate_reports_keeps_every_input_report(tmp_path):
    paths = [
        _write_report(tmp_path, "one.html", "session one"),
        _write_report(tmp_path, "two.html", "session two"),
    ]

    result = aggregate_reports.aggregate_reports(paths)
    contents = result.fspath.read_text(encoding="utf-8")

    assert "session one" in contents
    assert "session two" in contents
    assert "one.html" in contents and "two.html" in contents


def test_aggregate_reports_writes_to_given_output_path(tmp_path):
    paths = [_write_report(tmp_path, "one.html", "session one")]
    output = tmp_path / "out" / "combined.html"

    result = aggregate_reports.aggregate_reports(paths, output_path=output)

    assert result.fspath == output
    assert output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_aggregate_reports_accepts_string_paths(tmp_path):
    path = _write_report(tmp_path, "one.html", "session one")

    result = aggregate_reports.aggregate_reports([str(path)])

    assert "session one" in result.fspath.read_text(encoding="utf-8")


def test_aggregate_reports_with_no_inputs_still_returns_a_document(tmp_path):
    result = aggregate_reports.aggregate_reports([], output_path=tmp_path / "empty.html")

    assert isinstance(result, Html)
    assert "0 report(s) aggregated" in result.fspath.read_text(encoding="utf-8")


def test_aggregate_reports_raises_on_missing_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_reports.aggregate_reports([tmp_path / "absent.html"])


def test_load_reports_preserves_order(tmp_path):
    paths = [
        _write_report(tmp_path, "b.html", "second"),
        _write_report(tmp_path, "a.html", "first"),
    ]

    documents = aggregate_reports.load_reports(paths)

    assert "second" in documents[0]
    assert "first" in documents[1]


def test_combine_reports_escapes_source_labels(tmp_path):
    path = tmp_path / "<script>.html"

    combined = aggregate_reports.combine_reports(["<p>body</p>"], sources=[path])

    assert "&lt;script&gt;.html" in combined
    assert "<script>.html" not in combined
