import sys
sys.path.insert(0, r"J:\projet 1\tsauditor")
import pandas as pd
from tsauditor.leakage.equivalence import audit_equivalence
target = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] *4
feature_triche = [8, 2, 7, 1, 9, 3, 8, 2, 7, 1] *4
feature_normale = [100, 150, 120, 130, 90, 110, 140, 95, 105, 115] *4

df = pd.DataFrame({
         "target": target,
         "feature_triche": feature_triche,
         "feature_normale": feature_normale,
         })

issues = audit_equivalence(df, target="target")
print(len(issues))
print(issues[0].code)
print(issues[0].column)