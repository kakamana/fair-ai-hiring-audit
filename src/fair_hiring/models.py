"""Baseline + Fairlearn-postprocessed classifiers + audit table."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .features import build_feature_pipeline

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


def train_baseline(df: pd.DataFrame) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("pre", build_feature_pipeline(df).named_steps["pre"]),
            ("lr", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    y = df["hire_label"].values
    pipe.fit(df, y)
    return pipe


def train_postprocessed(df: pd.DataFrame, baseline: Pipeline, sensitive_col: str = "gender"):
    """Wrap baseline in fairlearn ThresholdOptimizer (equalized odds across gender)."""
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
    except ImportError:
        return None

    X = df
    y = df["hire_label"].values
    sensitive = df[sensitive_col].values

    to = ThresholdOptimizer(
        estimator=baseline,
        constraints="equalized_odds",
        prefit=True,
        predict_method="predict_proba",
    )
    to.fit(X, y, sensitive_features=sensitive)
    return to


def fairness_audit(
    df: pd.DataFrame, y_pred: np.ndarray, sensitive_col: str = "gender"
) -> pd.DataFrame:
    """Per-group selection rate, recall (TPR), FPR + the global gap."""
    rows = []
    y_true = df["hire_label"].values
    for grp, sub in df.groupby(sensitive_col):
        idx = sub.index.values
        yt = y_true[idx]
        yp = y_pred[idx]
        pos = max(int((yt == 1).sum()), 1)
        neg = max(int((yt == 0).sum()), 1)
        recall = float(((yp == 1) & (yt == 1)).sum() / pos)
        fpr = float(((yp == 1) & (yt == 0)).sum() / neg)
        sel_rate = float(yp.mean())
        rows.append(dict(group=grp, n=len(idx), recall=recall, fpr=fpr, selection_rate=sel_rate))
    out = pd.DataFrame(rows)
    out.attrs["recall_gap"] = float(out["recall"].max() - out["recall"].min())
    out.attrs["selection_rate_gap"] = float(out["selection_rate"].max() - out["selection_rate"].min())
    out.attrs["fpr_gap"] = float(out["fpr"].max() - out["fpr"].min())
    return out


def save(obj, name: str) -> Path:
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    return path


def load(name: str):
    return joblib.load(MODEL_DIR / name)


def main() -> None:
    p = DATA_PROC / "resumes.parquet"
    if not p.exists():
        from .data import make_dataset

        make_dataset()
    df = pd.read_parquet(p)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=11, stratify=df["hire_label"])
    baseline = train_baseline(train_df)
    save(baseline, "baseline_lr.joblib")
    pp = train_postprocessed(train_df, baseline, sensitive_col="gender")
    if pp is not None:
        save(pp, "fairlearn_eqodds.joblib")

    # Quick audit on the holdout
    yp_baseline = baseline.predict(test_df)
    audit_baseline = fairness_audit(test_df.reset_index(drop=True), yp_baseline)
    print("Baseline audit:")
    print(audit_baseline)
    print(
        f"  recall_gap={audit_baseline.attrs['recall_gap']:.3f} "
        f"selection_rate_gap={audit_baseline.attrs['selection_rate_gap']:.3f} "
        f"fpr_gap={audit_baseline.attrs['fpr_gap']:.3f}"
    )
    if pp is not None:
        yp_pp = pp.predict(test_df, sensitive_features=test_df["gender"].values)
        audit_pp = fairness_audit(test_df.reset_index(drop=True), yp_pp)
        print("Postprocessed audit:")
        print(audit_pp)


if __name__ == "__main__":
    main()
