from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURES

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="Trustworthy AI Audit API",version="0.1.0")


class RiskRequest(BaseModel):
    income: float = Field(gt=0)
    debt_ratio: float = Field(ge=0,le=1)
    history_months: float = Field(ge=0)
    late_payments: int = Field(ge=0)
    requested_amount: float = Field(gt=0)
    employment_years: float = Field(ge=0)


@lru_cache
def model():
    path=ROOT / "artifacts" / "model.joblib"
    if not path.exists(): raise FileNotFoundError("Run the audit pipeline first")
    return joblib.load(path)


@app.get("/health")
def health(): return {"status":"ok","model_ready":(ROOT/"artifacts"/"model.joblib").exists()}


@app.post("/assess")
def assess(request:RiskRequest):
    try:
        p=float(model().predict_proba(pd.DataFrame([request.model_dump()])[FEATURES])[:,1][0])
        action="human_review" if .4<=p<=.6 else ("high_risk" if p>.6 else "low_risk")
        return {"risk_probability":p,"action":action,"notice":"Synthetic portfolio demonstration; not for lending decisions."}
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail=str(exc)) from exc
