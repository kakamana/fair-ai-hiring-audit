# Methodology — Fair AI Hiring

## 1. Setup

Let the screening prediction be $\hat Y \in \{0, 1\}$, the true outcome $Y$, and the protected attribute $A$ (gender ∈ {Female, Male} in the default configuration).

## 2. Three fairness definitions you should know

### 2.1 Demographic parity (DP)

$$ \Pr(\hat Y = 1 \mid A = a) = \Pr(\hat Y = 1 \mid A = a') \quad \forall\, a, a' $$

— equal *selection rate* across groups, irrespective of true outcomes. Strong; can be violated even when the underlying outcome differs.

### 2.2 Equal opportunity (Hardt et al. 2016)

$$ \Pr(\hat Y = 1 \mid A = a, Y = 1) = \Pr(\hat Y = 1 \mid A = a', Y = 1) $$

— equal *true-positive rate* (recall) across groups. Symmetric in errors-on-positives.

### 2.3 Equalized odds (Hardt et al. 2016)

$$ \Pr(\hat Y = 1 \mid A = a, Y = y) = \Pr(\hat Y = 1 \mid A = a', Y = y), \quad y \in \{0, 1\} $$

— equal TPR *and* equal FPR. The constraint we enforce in this project.

## 3. Pre- vs in- vs post-processing

| Stage | Idea | Tool | Trade-off |
|-------|------|------|-----------|
| **Pre-processing** | Reweight / massage training data so a downstream learner produces a fair classifier | `Fairlearn.preprocessing` | Subject to data drift; needs retraining |
| **In-processing** | Add a fairness constraint to the optimization | `Fairlearn.reductions.ExponentiatedGradient` | Most flexible; costlier to fit |
| **Post-processing** | Adjust thresholds per group on top of an already-trained model | `Fairlearn.postprocessing.ThresholdOptimizer` | Lightweight; doesn't change training; chosen here |

We use **post-processing** because it leaves the underlying classifier unchanged and is cheapest to retrain — important for a screener that re-fits monthly.

## 4. ThresholdOptimizer with Equalized Odds

Given the baseline predictor $f$ and protected attribute $A$, ThresholdOptimizer learns group-specific thresholds $t_a$ — possibly with a randomization between two thresholds — such that

$$ \Pr(f(X) > t_A \mid A, Y) $$

is equalized across $A$ for $Y = 0$ and $Y = 1$ separately. The objective is to maximize accuracy subject to that constraint (Hardt et al. 2016, §4).

## 5. Audit

For every protected attribute audited, we report:

| Metric | Formula |
|--------|---------|
| Selection rate | $\Pr(\hat Y = 1 \mid A = a)$ |
| TPR (recall) | $\Pr(\hat Y = 1 \mid A = a, Y = 1)$ |
| FPR | $\Pr(\hat Y = 1 \mid A = a, Y = 0)$ |
| Max-min gap | $\max_a m_a - \min_a m_a$ for each metric |

The deployment gate: **all three gaps ≤ 5 percentage points** on the holdout.

## 6. Why this beats "just remove the protected attribute"

Dropping the protected attribute does *not* drop the proxies. In the synthetic dataset here, `prior_employer_tier` is correlated with both gender and the hire outcome. The audit is the only way to detect this; the post-processor is one way to fix it.

## 7. References
- Hardt, Price, Srebro (2016). *Equality of Opportunity in Supervised Learning*.
- Agarwal et al. (2018). *A Reductions Approach to Fair Classification*.
- Barocas, Hardt, Narayanan, *Fairness and Machine Learning*, fairmlbook.org.
- Fairlearn user guide, https://fairlearn.org/.
