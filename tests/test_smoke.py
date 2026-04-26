from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


SAMPLE = dict(
    cand_id="C-00001",
    years_experience=5,
    education_level="Bachelor",
    gender="Female",
    nationality_group="South Asian",
    prior_employer_tier=2,
    skill_tfidf_features=[0.1] * 32,
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_screen_stub():
    r = client.post("/screen", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["cand_id"] == "C-00001"
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in ("advance", "reject")
    assert body["fairness_postprocessed_decision"] in ("advance", "reject")
    assert "disclaimer" in body


def test_audit_endpoint():
    r = client.get("/audit", params={"sensitive": "gender"})
    assert r.status_code == 200
    body = r.json()
    # Either trained-bundle or stub-note path is acceptable here
    assert isinstance(body, dict)


def test_data_generator_shape():
    from fair_hiring.data import generate_resumes

    df = generate_resumes(n=400, seed=7)
    assert len(df) == 400
    expected = {"cand_id", "gender", "nationality_group", "prior_employer_tier", "hire_label"}
    assert expected.issubset(df.columns)
    assert df["hire_label"].isin([0, 1]).all()
    # Proxy bias should make Female candidates over-represented in tier 1
    fem_low = (df.query("gender == 'Female'")["prior_employer_tier"] == 1).mean()
    male_low = (df.query("gender == 'Male'")["prior_employer_tier"] == 1).mean()
    assert fem_low > male_low
