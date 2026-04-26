# Notebook 03 — Modeling

## 1. Baseline LR
>>> `from fair_hiring.models import train_baseline`
>>> `train_baseline(train_df)` — log accuracy + per-group metrics.

## 2. Audit (pre-mitigation)
>>> `fairness_audit(test_df, baseline.predict(test_df), 'gender')`
- Confirm gaps > 5 pts (the proxy bias is doing its job).

## 3. Fairlearn post-processing
>>> `train_postprocessed(train_df, baseline, sensitive_col='gender')`
- Equalized odds across gender.

## 4. Audit (post-mitigation)
>>> Confirm gaps ≤ 5 pts; report accuracy delta.

## 5. Persist
>>> `models.save(baseline, 'baseline_lr.joblib')`
>>> `models.save(pp, 'fairlearn_eqodds.joblib')`
