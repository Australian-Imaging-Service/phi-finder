from phi_finder.reporting import aggregate


def test_combine_reports_embeds_every_document():
    docs = [
        "<!DOCTYPE html><html><body><p>session one</p></body></html>",
        "<!DOCTYPE html><html><body><p>session two</p></body></html>",
    ]

    combined = aggregate._combine_reports(docs)

    assert isinstance(combined, str)
    assert combined.startswith("<!DOCTYPE html>")
    assert "2 report(s) aggregated" in combined
    assert "session one" in combined
    assert "session two" in combined


def test_combine_reports_uses_default_labels_when_none_given():
    combined = aggregate._combine_reports(["<p>a</p>", "<p>b</p>"])

    assert "Report 1" in combined
    assert "Report 2" in combined


def test_combine_reports_uses_and_escapes_given_labels():
    combined = aggregate._combine_reports(
        ["<p>body</p>"], labels=["session <1> & 2"]
    )

    # Labels are HTML-escaped so they cannot break out of the heading.
    assert "session &lt;1&gt; &amp; 2" in combined
    assert "session <1> & 2" not in combined


def test_combine_reports_with_no_documents_still_returns_a_document():
    combined = aggregate._combine_reports([])

    assert combined.startswith("<!DOCTYPE html>")
    assert "0 report(s) aggregated" in combined
