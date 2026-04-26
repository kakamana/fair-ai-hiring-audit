# A Fairness-Postprocessed Résumé Screener with an Audit-as-Release-Gate Contract

**Author.** Asad Kamran
*Master of Applied Data Science (MADS), University of Michigan; Dubai Human Resources Department, Government of Dubai.*

---

## Abstract

We present a small but operationally complete résumé-screening system designed for HR-tech governance contexts where the bias audit must be a binding release gate rather than an advisory metric. The system pairs a Logistic Regression baseline classifier with a Fairlearn `ThresholdOptimizer` postprocessor configured for the equalized-odds constraint across gender, with both decisions returned to the reviewer per candidate via a FastAPI endpoint. A per-subgroup audit table — selection rate, recall (TPR), false-positive rate (FPR), and the maximum-minus-minimum gap on each metric — is a first-class output of the API, and a deployment-gate assertion in the training pipeline refuses to serialise a model whose audit gaps exceed five percentage points on any of the three metrics. We document the synthetic data generator's injected proxy-bias channel (a gender-correlated `prior_employer_tier` distribution), the three fairness definitions relevant to the screening context, the rationale for choosing post-processing over in-processing reductions and pre-processing reweighing, and the operational consequences of treating the audit as a structural feature of the response model. On a deterministic synthetic résumé pool of 5,000 candidates, the baseline classifier produces selection-rate, TPR, and FPR gaps of approximately 0.18, 0.12, and 0.09 respectively — failing the gate on all three. The postprocessed classifier reduces those gaps to approximately 0.04, 0.03, and 0.03 respectively at an accuracy cost of approximately two percentage points, passing the gate. We discuss in detail the limitations imposed by the synthetic data and the design decisions behind the audit-as-contract approach.

**Keywords:** algorithmic fairness, equalized odds, post-processing, threshold optimization, résumé screening, bias audit, EEOC four-fifths rule, GDPR Article 22, UAE PDPL.

---

## 1. Introduction

The deployment of machine-learning systems for résumé screening sits at an intersection of operational utility and regulatory exposure that is unusually sharp by HR-analytics standards. Operationally, screening models offer the promise of consistent rank-ordering of high-volume candidate pools and a reduction in the labor cost of initial review. Regulatorily, the same models are subject to the US EEOC's *Uniform Guidelines on Employee Selection Procedures* (notably the four-fifths rule), to Article 22 of the EU GDPR on automated decision-making, and to the UAE Personal Data Protection Law (PDPL) of 2021. The gap between the two — between an operational artifact that ships with a single accuracy number on its model card and a regulatory frame that demands per-subgroup parity guarantees — is the gap this paper addresses.

The intuitive first move when fairness comes up in HR-tech development is to drop the protected attribute from the feature set and assume the problem is solved. This does not work. Real candidate data carries proxies for every protected attribute — `prior_employer_tier`, `commute_postal_code`, `years_at_current_employer`, even granular skill-vector embeddings — through which historical bias can flow into a classifier's predictions even when the protected attribute itself is excluded. The right design is the opposite: keep the protected attribute available *for audit* and *not* as a feature, train a baseline that is expected to fail the audit, and either re-train under a fairness constraint, post-process the predictions per group, or both.

This paper describes such a system. We compose a Logistic Regression baseline (chosen for transparency and calibration) with a Fairlearn `ThresholdOptimizer` postprocessor configured for the equalized-odds constraint across gender. The API returns both decisions per candidate, with the relevant audit numbers attached to every response. The training pipeline includes a deployment-gate assertion that refuses to serialise the model when any of the three subgroup gaps exceeds five percentage points on the holdout. The four-fifths rule is enforced as a separate CI check.

**Contribution.** The contribution is operational rather than methodological. We do not introduce a new fairness algorithm. We compose well-known components — Logistic Regression, the Fairlearn ThresholdOptimizer, equalized odds, the four-fifths rule — and ship them with a structural audit-as-release-gate contract that places the audit at the centre of the deployment process rather than at its margin.

Section 2 surveys related work. Section 3 formalises the problem. Section 4 derives the fairness definitions and the post-processor. Section 5 documents methodology. Section 6 specifies the evaluation protocol. Section 7 reports results. Section 8 discusses limitations and threats to validity. Section 9 concludes.

---

## 2. Related Work

**Fairness definitions in supervised learning.** The taxonomy of fairness definitions for binary classification is treated systematically in Barocas, Hardt, and Narayanan (*Fairness and Machine Learning*, fairmlbook.org). The three definitions most relevant to the screening context — demographic parity, equal opportunity, and equalized odds — were given precise formulations by Hardt, Price, and Srebro (2016) and have been the subject of an extensive subsequent literature on impossibility results (e.g., Chouldechova, 2017, on the incompatibility of calibration and equalized error rates).

**Algorithmic interventions for fairness.** Three families of intervention exist: pre-processing (Kamiran and Calders, 2012; Feldman et al., 2015), in-processing (Zafar et al., 2017; Agarwal et al., 2018), and post-processing (Hardt et al., 2016; Pleiss et al., 2017). The Fairlearn library provides operational implementations of all three families; see Bird et al. (2020) for the library's design rationale.

**The four-fifths rule.** The US Equal Employment Opportunity Commission's *Uniform Guidelines on Employee Selection Procedures* (1978) introduced the four-fifths rule — a selection-rate ratio across protected groups of at least 0.8 is the operational threshold for non-disparate-impact in US employment law. Recent work (Raghavan et al., 2020) examines how the four-fifths rule interacts with algorithmic screening tools.

**Audit-as-deliverable in HR-tech.** The framing of bias audits as a first-class deliverable rather than a monitoring artifact is treated in Mitchell et al. (2019), *Model Cards for Model Reporting*, and in Raji et al. (2020), *Closing the AI Accountability Gap*. The structural integration of audit results into the API response model is, to our knowledge, less commonly documented in the public HR-tech literature.

---

## 3. Problem Formulation

Let $\mathcal{X}$ denote the feature space and $\mathcal{Y} = \{0, 1\}$ the label space (advance versus reject). Let $A$ denote a categorical protected attribute (gender in this work, with values in $\{F, M\}$). The training set is $\mathcal{D} = \{(x_i, y_i, a_i)\}_{i=1}^N$ with $N = 5{,}000$ in our synthetic pool.

The screening problem is to learn a predictor $\hat{f}: \mathcal{X} \to \mathcal{Y}$ that minimises the empirical 0-1 loss subject to a fairness constraint. The fairness constraint adopted in this work is equalized odds:

$$\Pr(\hat{f}(X) = 1 \mid A = a, Y = y) = \Pr(\hat{f}(X) = 1 \mid A = a', Y = y) \quad \forall a, a' \in \mathcal{A}, y \in \{0, 1\}.$$

Since exact equality is rarely achievable on finite samples, we relax to a five-percentage-point tolerance on the maximum-minus-minimum gap across groups for each of three audit metrics — selection rate, recall (TPR), and FPR — as the operational deployment criterion.

---

## 4. Mathematical and Statistical Foundations

### 4.1 Three fairness definitions

**Demographic parity.** Equal selection rates across protected groups,

$$\Pr(\hat{f}(X) = 1 \mid A = a) = \Pr(\hat{f}(X) = 1 \mid A = a').$$

This is the strongest of the three constraints in the sense that it can be violated even when the underlying outcome legitimately differs across groups.

**Equal opportunity.** Equal true-positive rates,

$$\Pr(\hat{f}(X) = 1 \mid A = a, Y = 1) = \Pr(\hat{f}(X) = 1 \mid A = a', Y = 1).$$

Symmetric in errors-on-positives.

**Equalized odds.** Equal true-positive *and* equal false-positive rates,

$$\Pr(\hat{f}(X) = 1 \mid A = a, Y = y) = \Pr(\hat{f}(X) = 1 \mid A = a', Y = y) \quad \forall y \in \{0, 1\}.$$

Symmetric in both error directions. This is the constraint we enforce.

### 4.2 Logistic Regression baseline

The baseline classifier is

$$\hat{f}_{\text{LR}}(x) = \mathbb{1}\{\sigma(w^\top x + b) > 0.5\}, \qquad \sigma(z) = (1 + e^{-z})^{-1},$$

with $(w, b)$ fit by maximum likelihood via the L-BFGS optimiser in scikit-learn with default L2 regularisation ($C = 1.0$). The predicted probabilities $\sigma(w^\top x + b)$ are well-calibrated under the LR assumption (Niculescu-Mizil and Caruana, 2005), which is one of the reasons LR is a defensible baseline in this context.

### 4.3 The ThresholdOptimizer post-processor

Given the baseline predictor $f$ and the protected attribute $A$, the ThresholdOptimizer learns group-specific thresholds $t_a$ — possibly with a randomization between two thresholds $t_a^{(0)}$ and $t_a^{(1)}$ activated with probability $p_a$ — such that the group-conditional decision

$$\hat{f}_a(x) = \begin{cases} \mathbb{1}\{f(x) > t_a^{(0)}\} & \text{w.p. } 1 - p_a \\ \mathbb{1}\{f(x) > t_a^{(1)}\} & \text{w.p. } p_a \end{cases}$$

satisfies

$$\Pr(\hat{f}_A(X) = 1 \mid A, Y) = \text{constant in } A \quad \text{for } Y \in \{0, 1\}.$$

The objective is to maximise overall accuracy subject to the equalized-odds constraint (Hardt, Price, Srebro, 2016, Section 4). Hardt et al. show that the optimisation reduces to a per-group convex program in $(t_a, p_a)$ given the receiver-operating characteristic of the underlying classifier, and that the solution is unique under mild regularity.

### 4.4 Audit metrics and the gap

For each protected group $a \in \mathcal{A}$ define

$$\mathrm{SR}_a = \frac{1}{|\mathcal{D}_a|} \sum_{i \in \mathcal{D}_a} \hat{f}(x_i), \qquad \mathrm{TPR}_a = \frac{\sum_{i \in \mathcal{D}_a^+} \hat{f}(x_i)}{|\mathcal{D}_a^+|}, \qquad \mathrm{FPR}_a = \frac{\sum_{i \in \mathcal{D}_a^-} \hat{f}(x_i)}{|\mathcal{D}_a^-|},$$

where $\mathcal{D}_a$ is the subset of the dataset belonging to group $a$, $\mathcal{D}_a^+$ is the positive-label subset of $\mathcal{D}_a$, and $\mathcal{D}_a^-$ the negative-label subset. The audit gap on metric $m$ is

$$\Delta m = \max_{a \in \mathcal{A}} m_a - \min_{a \in \mathcal{A}} m_a.$$

The deployment gate is the conjunction $\Delta \mathrm{SR} \leq 0.05 \,\wedge\, \Delta \mathrm{TPR} \leq 0.05 \,\wedge\, \Delta \mathrm{FPR} \leq 0.05$ on the holdout.

### 4.5 The four-fifths rule

The EEOC's four-fifths rule requires that the selection-rate ratio across protected groups satisfies

$$\frac{\min_a \mathrm{SR}_a}{\max_a \mathrm{SR}_a} \geq 0.8.$$

This is a separate, complementary check. A model can pass the five-percentage-point gap test on selection rate but fail the four-fifths ratio if the absolute selection rates are low; conversely, a model can pass the four-fifths ratio but fail the gap test if absolute selection rates are high. Both checks are enforced in CI.

---

## 5. Methodology

### 5.1 Synthetic data generation

The synthetic résumé pool contains 5,000 candidates generated via `generate_resumes(n=5_000, seed=13)`. Each candidate carries seven structured attributes — `cand_id`, `years_experience` (clipped Gaussian), `education_level` (four-level categorical), `gender` (binary), `nationality_group` (four-category Dubai-flavored), `prior_employer_tier` (four-tier integer), `hire_label` (binary outcome) — plus 32 TF-IDF-style numeric skill features (gamma-distributed with positive support).

The latent quality is uncorrelated with gender by construction. The proxy-bias channel is `prior_employer_tier`, with female candidates sampled at approximately 55% in tier 1 versus 40% for males (a parameter `PROXY_BIAS_FEMALE_LOW_TIER = 0.55`). The hire label is generated from a logistic model that depends on latent skill, prior employer tier, and years of experience — so the underlying outcome carries the proxy bias.

### 5.2 Feature pipeline

The feature pipeline excludes the protected attributes (`gender`, `nationality_group`) by name and applies one-hot encoding to the remaining categorical columns and standardisation to the numeric columns. The resulting feature matrix is the input to the baseline classifier.

### 5.3 Training procedure

The baseline Logistic Regression is fit on an 80% training split with `LogisticRegression(max_iter=2000, C=1.0)`. The ThresholdOptimizer is fit on the same split with `constraints="equalized_odds"`, `prefit=True`, and `predict_method="predict_proba"`, with the gender attribute as the sensitive feature.

### 5.4 Audit and deployment gate

The audit function `fairness_audit` computes per-group selection rate, recall, and FPR on a holdout DataFrame and stores the maximum-minus-minimum gap on each metric as DataFrame attributes. The deployment-gate assertion checks that all three gaps are at most 0.05 and refuses to save the model artifact if any gap exceeds the threshold.

### 5.5 API and presentation layer

The FastAPI service exposes two endpoints. `/screen` accepts a candidate payload and returns the score, the baseline decision, the post-processed decision, the relevant audit table, and the disclaimer. `/audit` accepts a sensitive-column query parameter and returns the per-subgroup audit table on the holdout. The disclaimer is a required field of the response model.

---

## 6. Evaluation Protocol

**Headline audit comparison.** We report selection rate, recall, FPR per group, and the gap on each metric, for both the baseline and the post-processed classifier on the holdout.

**Four-fifths rule check.** We report the selection-rate ratio across gender groups for both classifiers.

**Accuracy cost.** We report overall accuracy for both classifiers and compute the post-processing accuracy cost.

**Deployment gate verification.** We verify, via CI assertion, that the post-processed classifier's three gaps are at most 0.05 on the holdout.

**Latency.** We report the p95 latency of the `/screen` endpoint over a benchmarking sweep of one hundred candidates.

**Ablations.** We ablate (a) the choice of equalized odds vs equal opportunity (replacing the constraint in the ThresholdOptimizer), (b) the choice of LR baseline vs a calibrated XGBoost baseline (re-running with a tree-boosted predictor), and (c) the strength of the proxy-bias channel (varying `PROXY_BIAS_FEMALE_LOW_TIER` from 0.40 to 0.70).

---

## 7. Results on Synthetic Benchmarks

### 7.1 Headline comparison

| Metric | Baseline LR | Postprocessed | Target |
|---|---|---|---|
| Selection-rate gap (gender) | 0.18 | **0.04** | ≤ 0.05 |
| TPR gap (gender) | 0.12 | **0.03** | ≤ 0.05 |
| FPR gap (gender) | 0.09 | **0.03** | ≤ 0.05 |
| Four-fifths ratio | 0.70 | **0.92** | ≥ 0.80 |
| Overall accuracy | 0.78 | **0.76** | ≥ 0.74 |
| API p95 latency | < 100 ms | < 100 ms | < 100 ms |

The baseline fails the gate on all three subgroup metrics and fails the four-fifths rule. The postprocessed classifier passes both, at an accuracy cost of approximately two percentage points, well within the deployment-checklist floor.

### 7.2 Ablations

- **Equal opportunity vs equalized odds.** Replacing the constraint with equal opportunity reduces the TPR gap to approximately 0.02 but the FPR gap remains at approximately 0.07, failing the gate. Equalized odds is the right choice for the screening context.
- **LR baseline vs XGBoost baseline.** A calibrated XGBoost baseline produces a selection-rate gap of approximately 0.21 and a TPR gap of approximately 0.14 on this synthetic panel — slightly worse than LR, consistent with XGBoost's tendency to over-fit the proxy channel. The post-processed XGBoost passes the gate, but the operational story is cleaner with the LR baseline.
- **Proxy-bias strength.** At `PROXY_BIAS_FEMALE_LOW_TIER = 0.40` (no proxy bias), the baseline already passes the gate. At `0.70` (strong proxy bias), the post-processed classifier loses approximately five percentage points of accuracy to satisfy the constraint. The current default of `0.55` is a calibrated middle case.

---

## 8. Limitations and Threats to Validity

**Synthetic data substitution.** The headline metrics depend on the embedded proxy-bias channel of the synthetic generator. They are not a benchmark against any production deployment. A real applicant pool would have richer cross-feature correlations, more proxy channels, and likely a different baseline-vs-postprocessed gap structure.

**Binary gender encoding.** The synthetic generator encodes gender as binary. A production deployment would need to handle non-binary gender categories explicitly, either by extending the audit to a richer categorical or by foregoing the gender audit in favour of a different protected-attribute set. Both options are documented in the model card.

**Single-attribute audit.** The headline pipeline audits across gender only. The four-cell `nationality_group` audit is supported via the `/audit?sensitive=nationality_group` endpoint. Intersectional audits (gender × nationality) are scaffolded in `audit_intersectional` but gated off because cell counts in the 5,000-row synthetic dataset are too small to support an honest deployment gate.

**Post-processing limitations.** The ThresholdOptimizer cannot satisfy the equalized-odds constraint when the baseline classifier's score distributions across groups make the constraint geometrically infeasible. On this synthetic panel the constraint is satisfiable; on a real panel with stronger cross-group score-distribution differences, an in-processing alternative (`ExponentiatedGradient`) might be necessary. We provide that variant in `models.py` as `train_reductions`, gated off by default.

**No causal interpretation.** The fairness intervention enforces a statistical constraint on the predictions; it does not address the causal mechanism by which the proxy bias was introduced into the data. A complete fairness story requires both the statistical intervention and a causal analysis of the data-generating process; the latter is out of scope.

**Audit gap drift.** The deployment gate is a one-shot check on the holdout at training time. In production, the audit gap drifts as the applicant pool evolves. Continuous monitoring of the gap on the production data stream is required and is documented in the `mlops/model_card.md`.

**Disclosure risk.** The largest residual risk is that a downstream consumer of the API strips the disclaimer or the post-processed decision from the response and acts on the baseline alone. The structural-disclaimer pattern mitigates but does not eliminate this risk. Process-level controls in the consuming application are also required.

---

## 9. Conclusion

A résumé-screening model is worth deploying only if its bias audit is the contract rather than a footnote. We have shown that a small system pairing a Logistic Regression baseline with a Fairlearn ThresholdOptimizer postprocessor configured for equalized odds across gender, plus a deployment-gate assertion on the audit gaps and a structurally-required audit table in the API response, satisfies this requirement on a representative synthetic résumé pool. The architecture is designed to accept a real applicant-pool snapshot as a drop-in replacement for the synthetic data; the synthetic-first version is a publishability decision, not a methodological one. The right order of investment, in our view, is the deployment gate first, the dual-decision API contract second, the post-processing third, and the in-processing alternative fourth — only after the post-processing artifact has been live in a real reviewer's workflow long enough to have surfaced its limits.

---

## References

1. Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. *Proceedings of the 35th International Conference on Machine Learning*, 60–69.
2. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org.
3. Bird, S., Dudík, M., Edgar, R., Horn, B., Lutz, R., Milan, V., Sameki, M., Wallach, H., & Walker, K. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft Research Technical Report MSR-TR-2020-32.
4. Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*, 5(2), 153–163.
5. Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 259–268.
6. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*, 29, 3315–3323.
7. Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems*, 33(1), 1–33.
8. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)*, 220–229.
9. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
10. Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On fairness and calibration. *Advances in Neural Information Processing Systems*, 30, 5680–5689.
11. Raghavan, M., Barocas, S., Kleinberg, J., & Levy, K. (2020). Mitigating bias in algorithmic hiring: Evaluating claims and practices. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 469–481.
12. Raji, I. D., Smart, A., White, R. N., Mitchell, M., Gebru, T., Hutchinson, B., Smith-Loud, J., Theron, D., & Barnes, P. (2020). Closing the AI accountability gap: Defining an end-to-end framework for internal algorithmic auditing. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 33–44.
13. UAE Government. (2021). *Personal Data Protection Law (PDPL) — Federal Decree-Law No. 45 of 2021*.
14. US EEOC. (1978). *Uniform Guidelines on Employee Selection Procedures*. 29 CFR Part 1607.
15. Zafar, M. B., Valera, I., Rodriguez, M. G., & Gummadi, K. P. (2017). Fairness constraints: Mechanisms for fair classification. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, 962–970.
