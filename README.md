# Fair AI Hiring — Résumé Screening with a Bias Audit

> **Résumé screening with explicit fairness constraints — a Logistic-Regression baseline plus a Fairlearn `ThresholdOptimizer` for equalized odds across gender, with a per-subgroup audit dashboard.**

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- Most screening models ship with a single number (accuracy / AUC). This one ships with a **fairness audit table** as a first-class output.
- A baseline classifier *plus* a Fairlearn post-processed classifier — both returned by the API, side by side. The audit shows which one you should actually trust per subgroup.
- Built for the "AI in HR" governance conversation that every people-team is now having.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — fairness definitions + post-processing](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) · [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) · [Source](./src/fair_hiring/) · [API](./api/main.py) · [UI](./ui/app/page.tsx)
- [CLAUDE.md](./CLAUDE.md) — paste prompt to resume in this folder

## Headline results (target)

| Metric | Baseline LR | Fairlearn (Equalized Odds) | Target |
|---|---|---|---|
| Selection-rate gap (gender) | 0.18 | **0.04** | ≤ 0.05 |
| TPR gap (gender) | 0.12 | **0.03** | ≤ 0.05 |
| FPR gap (gender) | 0.09 | **0.03** | ≤ 0.05 |
| Overall accuracy | 0.78 | **0.76** | ≥ 0.74 |

(*synthetic-data illustrative; populate after fit*)

## Quickstart

```bash
pip install -e ".[dev]"
python -m fair_hiring.data            # generate 5,000 synthetic résumés
python -m fair_hiring.models          # fit baseline LR + Fairlearn ThresholdOptimizer
jupyter lab notebooks/
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python · pandas · scikit-learn · **fairlearn** (ThresholdOptimizer, MetricFrame) · FastAPI · Next.js · Tailwind

## Author
Asad — MADS @ University of Michigan · Dubai HR
