# Notebook 01 — EDA: Synthetic résumé pool

>>> `from fair_hiring.data import generate_resumes`

## 1. Volume + label balance
- 5,000 résumés
- Hire-label rate overall and by gender / nationality

## 2. Proxy-bias detection
- `prior_employer_tier` distribution per gender — confirm Female over-sampled into tier 1
- Hire-rate by `prior_employer_tier` — confirm tier matters

## 3. Skill-feature distribution
- Histogram of a few skill_tfidf_XX columns
- Mean skill index per education_level
