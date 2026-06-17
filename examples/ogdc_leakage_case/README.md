# OGDC Leakage Case

The original motivating bug for tsauditor's leakage module. A same-day
percentage-change feature (`ChangeP`) was mathematically near-identical to the
direction target it was meant to predict.

## Run it

python compare_leakage.py

Requires `ogdc_with_regimes.csv` (included in this folder) and scikit-learn.

## Output

`leakage_comparison_results.csv` — accuracy and AUC for Random Forest and GBM
classifiers, with and without the leaky features (`Open`, `High`, `Low`, `ChangeP`).
Last verified run: 99.68% accuracy with leakage, dropping to 69.81%–73.70% without it.