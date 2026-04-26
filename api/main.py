"""FastAPI for Fair AI Hiring.

Endpoints:
    GET  /health
    POST /screen   - score one candidate; return baseline + post-processed decision
    GET  /audit    - per-subgroup fairness audit table (gender by default)
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fair_hiring.serve import full_audit, screen

app = FastAPI(title="Fair AI Hiring", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "Screening scores are decision-support for HR / TA. The fairness-postprocessed "
    "decision should be the binding one when the audit gap exceeds 5 pts."
)


class CandidatePayload(BaseModel):
    cand_id: str = Field(..., description="Candidate reference")
    years_experience: int = Field(..., ge=0, le=60)
    education_level: Literal["High School", "Bachelor", "Master", "PhD"]
    gender: Literal["Female", "Male"]
    nationality_group: Literal["Emirati", "South Asian", "Western", "Other"] = "Other"
    prior_employer_tier: int = Field(..., ge=1, le=4)
    skill_tfidf_features: list[float] = Field(
        default_factory=lambda: [0.0] * 32,
        description="32 skill TF-IDF-style numeric features",
    )


class ScreenResponse(BaseModel):
    cand_id: str
    decision: Literal["advance", "reject"]
    score: float
    fairness_postprocessed_decision: Literal["advance", "reject"]
    audit: dict
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/screen", response_model=ScreenResponse)
def screen_endpoint(payload: CandidatePayload) -> ScreenResponse:
    try:
        out = screen(payload.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return ScreenResponse(**out)


@app.get("/audit")
def audit(sensitive: str = Query("gender")) -> dict:
    try:
        return full_audit(sensitive_col=sensitive)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
