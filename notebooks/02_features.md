# Notebook 02 — Feature pipeline

>>> `from fair_hiring.features import build_feature_pipeline, skill_columns`

## 1. Build the preprocessor
>>> `pipe = build_feature_pipeline(df)`

## 2. Inspect output shape
>>> `pipe.fit_transform(df).shape`

## 3. Sanity
- StandardScaler on numeric + skill features
- One-hot on `education_level`
