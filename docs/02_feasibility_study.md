# Feasibility Study — Fair AI Hiring

## 1. Data feasibility
- **Synthetic generator** (this repo) is enough to demonstrate the audit + post-processing flow end-to-end.
- **Real-data path:** plug an internal applicant-pool snapshot (anonymized, no free-text résumés in v0.1) — the pipeline is column-name compatible.

## 2. Technical feasibility
- **Algorithmic shortlist**
  - Baseline: Logistic Regression (calibrated by default in scikit-learn for binary tasks)
  - Post-processor: Fairlearn `ThresholdOptimizer` with `equalized_odds`
  - Future variant: Reductions approach via `ExponentiatedGradient` for in-processing
- **Compute:** seconds on 1 CPU; the dataset is < 100 MB.
- **Serving:** FastAPI + a 2-artifact joblib bundle (~1 MB total).

## 3. Economic feasibility
| Line item | Monthly cost |
|-----------|--------------|
| 1× small container | ~$8 |
| **Total** | **~$8 / mo** |

## 4. Operational feasibility
- **Refresh cadence:** per applicant-pool refresh (typically monthly).
- **Monitoring:** subgroup gap tracked in CI; fail the build if any gap > 5 pts on the holdout.
- **Human-in-the-loop:** every advance/reject decision is presented for HR review with the audit-adjusted decision shown side by side.

## 5. Ethical / legal feasibility
- **Sensitive attributes** used only for audit and post-processing — never as a feature.
- **Disclaimer** on every API response.
- **EEOC 4/5 rule:** selection-rate ratio across gender groups must be ≥ 0.8 — checked in CI.
- **UAE PDPL / EU GDPR-equivalent right-to-explanation:** every decision returns the score + the audit-adjusted decision.

## 6. Recommendation
**Go.** Cheap, defensible, and the dual-decision API contract makes the governance story easy to tell.
