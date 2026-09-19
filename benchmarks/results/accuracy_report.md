# tsauditor leakage-detection accuracy benchmark

Measures `report.leaky_columns()` precision/recall against real datasets whose leaky columns are independently documented (see each case's Source, not synthetic ground truth and not a leak injected for this benchmark). Regenerate with `python benchmarks/accuracy_suite.py`.

## ogdc_changep

OGDC (Oil & Gas Development Company Limited), Pakistan Stock Exchange daily OHLCV plus engineered features. tsauditor's own original motivating case.

Two different leakage *mechanisms* are mixed in this one source, deliberately not smoothed over: ChangeP and Returns are statistical target-equivalence (LEK001) -- detectable from the values alone, no extra information needed. Open, High, and Low are leaky for a different reason entirely: they are same-day quantities not known until the trading session closes, a point-in-time fact about *when* data becomes available that cannot be inferred from the values themselves -- that is exactly what LEK004 exists for, and it requires the caller to supply that fact via `available_at`. Run without it, only 2/5 are caught (LEK001 alone); with it, all 5/5. This case runs with `available_at` supplied for that reason -- omitting it would not be a harder test, it would be an unfair one: it silently omits the one piece of information LEK004's contract explicitly asks the caller to provide.

**Source:** examples/ogdc_leakage_case/README.md, build_ogdc_notebook.py, and compare_leakage.py (this repository): Open, High, Low, ChangeP, and Returns are all same-day quantities not available at prediction time; ChangeP and Returns additionally, mathematically define Direction's sign. Measured impact of removing all five: Random Forest accuracy 99.68% -> 69.81%, Gradient Boosting 99.68% -> 73.70%.

- Rows: 1537
- Recall: 100.0% (5/5 known-leaky columns caught: ['ChangeP', 'High', 'Low', 'Open', 'Returns'])
- Precision: 100.0%
- False positive rate: 0.0% (of 18 known-clean columns)

## ogdc_changep_statistical_only

Same OGDC data and same five-column ground truth as ogdc_changep above, but *without* `available_at` -- the baseline a caller gets if they scan without supplying point-in-time metadata (the common case: most callers won't have publish-timing metadata handy for every column). Reported side by side with the case above specifically so the LEK001-alone number (statistically detectable leaks) isn't hidden behind the LEK001+LEK004 combined number.

**Source:** Same as ogdc_changep.

- Rows: 1537
- Recall: 40.0% (2/5 known-leaky columns caught: ['ChangeP', 'Returns'])
  - **Missed:** ['High', 'Low', 'Open']
- Precision: 100.0%
- False positive rate: 0.0% (of 18 known-clean columns)

## alfred_gdp_revision

Real US GDP (GDPC1) from ALFRED (Archival FRED, Federal Reserve Bank of St. Louis) -- the standard real-time-data tool in the macro-forecasting literature (Croushore & Stark and others) for exactly this leak: a later data revision used as though it were available at the time. Two vintages of the same 191 quarters (1947-1994): as known in the 1995-01-30 vintage (legitimately available then) and as later revised in the 2024-01-01 vintage (not knowable for decades). The two are highly correlated (Pearson r = 0.998 over this window) -- GDP revisions are typically modest relative to level -- which makes this a harder, more realistic leak than a near-duplicate column: it could plausibly read as 'a slightly different measurement of the same thing' rather than a temporal-availability violation, unless the check actually reasons about *when* each column became known.

**Source:** ALFRED download, series GDPC1, 'Observations by Vintage Date, All Observations', vintages 1995-01-01 through 2024-01-01 (benchmarks/data/gdpc1_vintages.csv). The 2024-01-01 column's actual vintage-release date -- a fact from the download itself, not chosen for this benchmark -- is what `available_at` below is built from.

- Rows: 191
- Recall: 100.0% (1/1 known-leaky columns caught: ['gdp_level_asrevised2024'])
- Precision: 100.0%
- False positive rate: 0.0% (of 1 known-clean columns)

## alfred_gdp_revision_statistical_only

Same ALFRED data and same ground truth as alfred_gdp_revision above, but without `available_at` -- same before/after pairing as ogdc_changep_statistical_only, for the same reason: shows the floor a caller gets without supplying the one fact (publish timing) no statistical test can infer from values alone.

**Source:** Same as alfred_gdp_revision.

- Rows: 191
- Recall: 0.0% (0/1 known-leaky columns caught: [])
  - **Missed:** ['gdp_level_asrevised2024']
- Precision: n/a
- False positive rate: 0.0% (of 1 known-clean columns)
