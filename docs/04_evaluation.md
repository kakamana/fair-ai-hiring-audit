# Evaluation Plan — Fair AI Hiring

## 1. Held-out test set
20% stratified holdout on `hire_label`.

## 2. Primary scorecard

| Model | Accuracy | Sel-rate gap (gender) | TPR gap | FPR gap | 4/5 ratio |
|-------|----------|-----------------------|---------|---------|-----------|
| Baseline LR | – | – | – | – | – |
| Fairlearn (ThresholdOptimizer, EqOdds) | – | – | – | – | – |

**Deployment gate:** all three gender gaps ≤ 5 pts AND 4/5 ratio ≥ 0.8 on the post-processed model.

## 3. Subgroup audits

For each protected attribute (`gender`, `nationality_group`):
- Selection rate, TPR, FPR per subgroup
- Max-min gap and the equalized-odds difference
- Confusion matrix per subgroup

## 4. Calibration check

- Reliability diagram on the baseline model (per gender group).
- Brier score per group.

## 5. Sensitivity / robustness

- Vary the proxy-bias parameter `PROXY_BIAS_FEMALE_LOW_TIER` ∈ {0.40, 0.50, 0.55, 0.65}.
- Confirm the audit detects the disparity and the post-processor closes it across the range.
- Drop `prior_employer_tier` entirely — confirm fairness gap shrinks but baseline accuracy drops more than the fair model's (the fair model is "wrong on purpose" in the bad equilibrium).

## 6. Comparative analysis

Add a Reductions-approach (`ExponentiatedGradient`) baseline for context — same constraint, different mechanism.

## 7. Deployment readiness checklist
- [ ] All three gender gaps ≤ 5 pts on post-processed model
- [ ] 4/5 ratio ≥ 0.8
- [ ] Both decisions returned per candidate
- [ ] `/audit` exposes the full table
- [ ] Model card published at `mlops/model_card.md`
- [ ] Disclaimer present on every API response and in the UI
