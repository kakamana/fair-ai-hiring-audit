"""Inference helpers used by the FastAPI layer."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import models

DATA_PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


@lru_cache(maxsize=1)
def _load() -> dict:
    out: dict = {}
    try:
        out["baseline"] = models.load("baseline_lr.joblib")
    except FileNotFoundError:
        out["baseline"] = None
    try:
        out["fairlearn"] = models.load("fairlearn_eqodds.joblib")
    except FileNotFoundError:
        out["fairlearn"] = None
    return out


def _featurize_one(payload: dict) -> pd.DataFrame:
    """Coerce a single API payload into a one-row DataFrame in training shape."""
    skill_features = payload.pop("skill_tfidf_features", None)
    if skill_features is None:
        skill_features = [0.0] * 32
    base = dict(payload)
    for j, v in enumerate(skill_features):
        base[f"skill_tfidf_{j:02d}"] = float(v)
    return pd.DataFrame([base])


def _stub_score(payload: dict) -> float:
    """Deterministic-ish score so the API is useful before training has run."""
    yrs = float(payload.get("years_experience", 0))
    tier = float(payload.get("prior_employer_tier", 1))
    return float(min(0.95, max(0.05, 0.10 + 0.04 * yrs + 0.10 * tier)))


def screen(payload: dict) -> dict:
    art = _load()
    cand_id = payload.get("cand_id", "C-XXXXX")
    sensitive = payload.get("gender", "Unknown")
    X = _featurize_one(dict(payload))

    base = art["baseline"]
    pp = art["fairlearn"]

    if base is None:
        score = _stub_score(payload)
        decision = "advance" if score >= 0.5 else "reject"
        return dict(
            cand_id=cand_id,
            decision=decision,
            score=score,
            fairness_postprocessed_decision=decision,
            audit=dict(note="model not trained yet — using stub"),
        )

    score = float(base.predict_proba(X)[0, 1])
    base_decision = "advance" if score >= 0.5 else "reject"

    if pp is None:
        pp_decision = base_decision
    else:
        try:
            pp_pred = pp.predict(X, sensitive_features=np.array([sensitive]))
            pp_decision = "advance" if int(pp_pred[0]) == 1 else "reject"
        except Exception:
            pp_decision = base_decision

    return dict(
        cand_id=cand_id,
        decision=base_decision,
        score=score,
        fairness_postprocessed_decision=pp_decision,
        audit=dict(sensitive_feature=sensitive),
    )


def full_audit(sensitive_col: str = "gender") -> dict:
    """Compute the per-subgroup audit table on the held-out resume panel."""
    p = DATA_PROC / "resumes.parquet"
    if not p.exists():
        return dict(rows=[], note="run `python -m fair_hiring.data` first")
    df = pd.read_parquet(p)
    art = _load()
    base = art["baseline"]
    pp = art["fairlearn"]
    if base is None:
        return dict(rows=[], note="run `python -m fair_hiring.models` first")
    y_pred_base = base.predict(df)
    audit_base = models.fairness_audit(df.reset_index(drop=True), y_pred_base, sensitive_col)
    out = dict(
        baseline=dict(
            rows=audit_base.to_dict(orient="records"),
            recall_gap=audit_base.attrs["recall_gap"],
            selection_rate_gap=audit_base.attrs["selection_rate_gap"],
            fpr_gap=audit_base.attrs["fpr_gap"],
        )
    )
    if pp is not None:
        try:
            y_pred_pp = pp.predict(df, sensitive_features=df[sensitive_col].values)
            audit_pp = models.fairness_audit(df.reset_index(drop=True), y_pred_pp, sensitive_col)
            out["fairlearn"] = dict(
                rows=audit_pp.to_dict(orient="records"),
                recall_gap=audit_pp.attrs["recall_gap"],
                selection_rate_gap=audit_pp.attrs["selection_rate_gap"],
                fpr_gap=audit_pp.attrs["fpr_gap"],
            )
        except Exception as exc:  # noqa: BLE001
            out["fairlearn_error"] = str(exc)
    return out
