"""Synthetic résumé dataset for the fairness-constrained hiring project.

5,000 candidates, with:
    cand_id, years_experience, education_level, gender, nationality_group,
    prior_employer_tier, hire_label, plus 32 TF-IDF-style numeric "skill" features.

A mild *proxy bias* is injected: the model's true latent quality is uncorrelated
with gender, but `prior_employer_tier` is correlated with both gender (proxy) and
the hire label — exactly the failure mode a fairness audit should catch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"

N_CANDIDATES = 5_000
N_SKILL_FEATURES = 32

GENDERS = ["Female", "Male"]
NATIONALITIES = ["Emirati", "South Asian", "Western", "Other"]
NATIONALITY_P = [0.12, 0.55, 0.15, 0.18]
EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "PhD"]
EDUCATION_P = [0.10, 0.55, 0.30, 0.05]

# Proxy-bias parameter: prob that a Female candidate's prior employer is "lower tier"
PROXY_BIAS_FEMALE_LOW_TIER = 0.55  # vs ~0.40 for males — mild but detectable


def generate_resumes(n: int = N_CANDIDATES, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(GENDERS, size=n, p=[0.45, 0.55])
    nationality = rng.choice(NATIONALITIES, size=n, p=NATIONALITY_P)
    education = rng.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_P)
    years_experience = np.clip(rng.normal(7, 4, size=n), 0, 35).astype(int)

    # Education-level numeric proxy
    edu_num = np.array([EDUCATION_LEVELS.index(e) for e in education])

    # prior_employer_tier 1..4 (4 = top), with proxy bias against Female applicants
    base_tier_probs = np.array([0.20, 0.30, 0.30, 0.20])
    tier = np.zeros(n, dtype=int)
    for i in range(n):
        p = base_tier_probs.copy()
        if gender[i] == "Female":
            shift = PROXY_BIAS_FEMALE_LOW_TIER - 0.40
            # shift mass from tier 4 to tier 1 by `shift`
            p[0] += shift
            p[3] -= shift
            p = np.clip(p, 1e-3, None)
            p = p / p.sum()
        tier[i] = int(rng.choice([1, 2, 3, 4], p=p))

    # Latent quality — uncorrelated with gender, depends on edu / experience / skills
    latent_skill_score = rng.normal(0.5 * edu_num + 0.05 * years_experience, 1.0)

    # 32 "skill" features — sparse positive (TF-IDF-like)
    skills_matrix = rng.gamma(0.8, 0.6, size=(n, N_SKILL_FEATURES))
    # weight a few features by latent skill
    weights = np.zeros(N_SKILL_FEATURES)
    weights[: max(1, N_SKILL_FEATURES // 4)] = 1.0
    skills_matrix = skills_matrix + np.outer(latent_skill_score, weights) * 0.05
    skills_matrix = np.maximum(skills_matrix, 0.0)

    # Hire label: depends on latent skill + tier (the proxy-bias channel)
    logit = (
        -0.7
        + 0.55 * latent_skill_score
        + 0.45 * (tier - 2.5)         # higher tier => more likely to be hired
        + 0.04 * years_experience
    )
    p_hire = 1.0 / (1.0 + np.exp(-logit))
    hire_label = rng.binomial(1, p_hire)

    df = pd.DataFrame(
        dict(
            cand_id=[f"C-{i:05d}" for i in range(1, n + 1)],
            years_experience=years_experience,
            education_level=education,
            gender=gender,
            nationality_group=nationality,
            prior_employer_tier=tier,
            hire_label=hire_label,
        )
    )
    skill_cols = [f"skill_tfidf_{j:02d}" for j in range(N_SKILL_FEATURES)]
    skills_df = pd.DataFrame(skills_matrix, columns=skill_cols)
    return pd.concat([df, skills_df], axis=1)


def make_dataset() -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = generate_resumes()
    df.to_parquet(PROCESSED / "resumes.parquet", index=False)
    return df


if __name__ == "__main__":
    df = make_dataset()
    print(
        f"wrote {len(df):,} résumés -> data/processed/resumes.parquet "
        f"(positive rate = {df['hire_label'].mean():.3f})"
    )
