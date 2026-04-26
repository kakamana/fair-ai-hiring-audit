# Data Card — H17 Fair AI Hiring

## Dataset composition

| Layer | Source | Rows × cols | Purpose |
|-------|--------|-------------|---------|
| Synthetic résumés | `src/fair_hiring/data.py::generate_resumes` | 5,000 × ~40 | Hiring screening + bias-audit benchmark |

## Fields
| Field | Type | Notes |
|-------|------|-------|
| `cand_id` | string | `C-NNNNN` |
| `years_experience` | int | 0 – 35 |
| `education_level` | category | High School / Bachelor / Master / PhD |
| `gender` | category | Female / Male (binary by source-data convention) |
| `nationality_group` | category | Emirati / South Asian / Western / Other |
| `prior_employer_tier` | int 1–4 | Proxy-bias channel — see "Known biases" |
| `skill_tfidf_XX` | float (×32) | Sparse, gamma-distributed numeric features |
| `hire_label` | int 0/1 | Outcome label |

## Known biases (intentional)
- **Proxy bias via `prior_employer_tier`**: Female candidates are over-sampled into tier 1 ("lower") at ~55% vs ~40% for males. The label depends on tier *and* latent skill.
- This is exactly the failure mode the audit + Fairlearn post-processor should catch.

## PII
None — synthetic, no real candidates.

## Splits
80% train / 20% holdout, stratified on `hire_label`.

## Reproducing
```bash
python -m fair_hiring.data
```
Deterministic seed = 13.

## Licensing
- Code: MIT
- Synthetic data: MIT (this repo)
