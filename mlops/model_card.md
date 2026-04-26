# Model Card — Fair AI Hiring

## Intended use
Decision-aid for résumé screening — returns a baseline score *and* a Fairlearn-postprocessed (equalized-odds) decision side by side. The post-processed decision is binding when the audit gap exceeds 5 pts.

## Training data
Synthetic 5,000-résumé panel with an injected proxy-bias channel (`prior_employer_tier`); see `data/data_card.md`.

## Model family
- Baseline: `sklearn.linear_model.LogisticRegression`
- Post-processor: `fairlearn.postprocessing.ThresholdOptimizer` with `equalized_odds` constraint, prefit baseline

## Metrics (to populate)
| Metric | Target |
|--------|--------|
| Selection-rate gap (gender) | ≤ 0.05 |
| TPR gap (gender) | ≤ 0.05 |
| FPR gap (gender) | ≤ 0.05 |
| Accuracy (post-processed) | ≥ 0.74 |
| 4/5 ratio | ≥ 0.80 |
| API p95 latency | < 100 ms |

## Limitations
- Synthetic data; bias channel injected by design.
- Binary gender encoding from source.
- Single-attribute audit by default — intersectional audits are flagged as future work.

## Ethical considerations
- Sensitive attributes used **only** for audit + post-processing, never as a feature for the baseline.
- Both decisions returned per request — the system never hides which one was used.
- Disclaimer in every API response and in the UI.

## Retraining
- Per applicant-pool refresh; CI fails if any gap > 5 pts.
