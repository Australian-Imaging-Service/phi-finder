import pytest
from fileformats.text.unicode import Html

from phi_finder.reporting import aggregate


def _write_report(directory, name, body):
    path = directory / name
    path.write_text(
        f"<!DOCTYPE html><html><body><p>{body}</p></body></html>", encoding="utf-8"
    )
    return path


def test_aggregate_returns_html_fileformat(tmp_path):
    paths = [
        _write_report(tmp_path, "one.html", "session one"),
        _write_report(tmp_path, "two.html", "session two"),
    ]

    result = aggregate.aggregate_reports(paths)

    assert isinstance(result, Html)
    assert result.fspath.exists()
    assert result.fspath.suffix == ".html"


def test_aggregate_writes_to_given_output_path(tmp_path):
    paths = [_write_report(tmp_path, "one.html", "session one")]
    output = tmp_path / "out" / "combined.html"

    result = aggregate.aggregate_reports(paths, output_path=output)

    assert result.fspath == output
    assert output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_aggregate_with_no_inputs_still_returns_a_document(tmp_path):
    result = aggregate.aggregate_reports([], output_path=tmp_path / "empty.html")

    assert isinstance(result, Html)
    assert "0 report(s) aggregated" in result.fspath.read_text(encoding="utf-8")


def test_aggregate_raises_on_missing_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate.aggregate_reports([tmp_path / "absent.html"])

