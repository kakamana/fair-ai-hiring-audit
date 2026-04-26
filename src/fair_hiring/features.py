"""Feature pipeline for résumé screening."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["years_experience", "prior_employer_tier"]
CATEGORICAL_FEATURES = ["education_level"]


def skill_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("skill_tfidf_")]


def build_feature_pipeline(df: pd.DataFrame) -> Pipeline:
    """Numeric scaler + one-hot for education + pass-through skills features."""
    skills = skill_columns(df)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + skills),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("pre", pre)])
