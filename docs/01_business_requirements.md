# Business Requirements — Fair AI Hiring

## 1. Problem Statement
Most production résumé screeners are trained on hiring outcomes that already encode historical bias. A model that is "accurate" on those outcomes can still produce disparate selection rates across protected groups. The HR governance question is no longer *can we screen with AI?* — it is *how do we screen with AI in a way we can defend in front of a regulator and an EEO officer?*

This project ships a working baseline (Logistic Regression) **plus** a Fairlearn `ThresholdOptimizer` post-processed classifier that enforces equalized odds across gender. Both decisions are returned by the API. The audit gap drives which one becomes binding.

## 2. Stakeholders
| Role | Interest | Success criterion |
|------|----------|-------------------|
| Talent Acquisition | Faster, defensible screening | Per-candidate response includes the audit-adjusted decision |
| HR / D&I lead | Subgroup parity guarantees | TPR / FPR / selection-rate gaps ≤ 5 pts |
| Legal / Compliance | EEOC 4/5-rule defensibility | Selection-rate ratio ≥ 0.8 across gender groups |
| Data Protection | Sensitive-attribute handling | Sensitive features used only for audit + post-processing |

## 3. Business Objectives
1. Train a baseline LR + a Fairlearn-postprocessed classifier on the synthetic résumé panel.
2. Surface a subgroup audit table (selection rate, TPR, FPR + max gap) on demand via `/audit`.
3. Per-candidate `/screen` returns *both* decisions — baseline and post-processed — so reviewers can see when they diverge.
4. Document the fairness definitions used; cite Hardt et al. and the Fairlearn user guide.

## 4. KPIs
| KPI | Target |
|-----|--------|
| Subgroup selection-rate gap (gender) | ≤ 5 pts |
| Subgroup TPR gap (gender) | ≤ 5 pts |
| Subgroup FPR gap (gender) | ≤ 5 pts |
| Overall accuracy (post-processing) | ≥ 0.74 |
| API p95 latency (single screen) | < 100 ms |

## 5. Scope
**In:** binary gender + 4-group nationality audit; logistic-regression baseline; ThresholdOptimizer post-processor.
**Out:** intersectional audits (gender × nationality), reductions-approach (Exp Gradient) variants, deep models on raw text — all are flagged as future work.

## 6. Constraints & Assumptions
- Synthetic data only in the public repo.
- Gender encoded binary in source data; the audit operates at that granularity.
- The disclaimer is on every API response.

## 7. Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Treating fairness as a single number | High | High | Audit reports three metrics, not one |
| Hiding bias under the hood | High | Critical | Both decisions returned per candidate |
| Post-processing trades too much accuracy | Medium | Medium | Accuracy floor in the deployment checklist |
| Sensitive attributes leaked elsewhere in the system | Medium | High | API contract restricts sensitive use to audit + post-processing |
