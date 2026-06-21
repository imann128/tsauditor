from tsauditor.report.summary import Issue, GuardReport, CRITICAL, WARNING, INFO


def test_issue_suggestion_mentions_column_and_action():
    i = Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP", {"separation": 1.0})
    s = i.suggestion
    assert "ChangeP" in s and ("Remove" in s or "reconstruct" in s)


def test_lek002_suggestion_fills_peak_lag():
    i = Issue("leakage", "LEK002", WARNING, "x", "leak", {"peak_lag": 1})
    assert "+1" in i.suggestion


def test_dataset_level_issue_says_dataset():
    i = Issue("profiler", "PRF001", WARNING, "irregular", None, {})
    assert "the dataset" in i.suggestion


def test_unknown_code_falls_back():
    i = Issue("x", "ZZZ999", INFO, "?", None, {})
    assert i.suggestion and "Review this issue" in i.suggestion


def test_missing_placeholder_does_not_crash():
    # LEK002's template references {peak_lag}; if evidence lacks it the
    # suggestion must still render rather than raising KeyError.
    i = Issue("leakage", "LEK002", WARNING, "x", "leak", {})
    assert isinstance(i.suggestion, str) and i.suggestion


def test_to_dict_includes_suggestion():
    d = Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP", {}).to_dict()
    assert "suggestion" in d and d["code"] == "LEK001" and d["column"] == "ChangeP"


def test_leaky_columns_lists_only_leakage_columns():
    r = GuardReport(
        critical=[Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP")],
        warnings=[
            Issue("leakage", "LEK002", WARNING, "x", "RSI"),
            Issue("profiler", "PRF003", WARNING, "ns", "Price"),
        ],
    )
    assert r.leaky_columns() == ["ChangeP", "RSI"]


def test_suggestions_structure_and_severity_order():
    r = GuardReport(
        critical=[Issue("leakage", "LEK001", CRITICAL, "eq", "ChangeP")],
        warnings=[Issue("profiler", "PRF003", WARNING, "ns", "Price")],
    )
    sg = r.suggestions()
    assert [s["severity"] for s in sg] == ["critical", "warning"]
    assert all({"code", "column", "severity", "suggestion"} <= set(s) for s in sg)


def test_empty_report_has_no_suggestions_or_leaky_columns():
    assert GuardReport().leaky_columns() == []
    assert GuardReport().suggestions() == []
