"""
Bias Breakers — FastAPI Backend v2.0
Run: uvicorn main:app --reload
"""

import io
import os
import json
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from bias_engine import BiasEngine

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bias Breakers",
    version="2.0.0",
    description="AI-Powered Bias Detection & Fairness Auditing Platform"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = BiasEngine()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_dataset() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "datasets", "indiahire_bias.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Dataset missing. Run:  python generate_dataset.py"
        )
    return pd.read_csv(path)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="index.html not found in static/")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "project": "Bias Breakers",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "engine": "BiasEngine v2.0 — Real ML Pipeline"
    }


@app.get("/api/demo")
async def run_demo():
    df     = _load_dataset()
    result = engine.full_analysis(df)
    return result


@app.post("/api/predict")
async def predict_student(student: dict):
    """
    Accepts a single student dict, returns biased decision, fair decision,
    rejection reasons, bias factors, and improvement suggestions.
    """
    df     = _load_dataset()
    result = engine.predict_student(student, df)
    return result


@app.post("/api/compare")
async def compare_scenarios(payload: dict = Body(...)):
    """
    Compare two student profiles side-by-side to expose bias.
    Expects: { "profile_a": {...}, "profile_b": {...} }
    """
    profile_a = payload.get("profile_a")
    profile_b = payload.get("profile_b")

    if not profile_a or not profile_b:
        raise HTTPException(
            status_code=422,
            detail="Both profile_a and profile_b are required."
        )

    df = _load_dataset()
    result = engine.compare_scenarios(profile_a, profile_b, df)
    return result


@app.post("/api/batch")
async def batch_predict(payload: dict = Body(...)):
    """
    Predict outcomes for multiple students at once.
    Expects: { "students": [{...}, {...}, ...] }
    """
    students = payload.get("students", [])
    if not students:
        raise HTTPException(status_code=422, detail="No students provided.")
    if len(students) > 20:
        raise HTTPException(status_code=422, detail="Maximum 20 students per batch.")

    df = _load_dataset()
    result = engine.batch_predict(students, df)
    return result


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        raw = await file.read()
        df  = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the CSV.")

    for col in ("hiring_decision", "gender"):
        if col not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Missing column '{col}'. "
                       "CSV must contain: gender (0=Female/1=Male), hiring_decision (0/1)."
            )

    result = engine.full_analysis(df)
    return result


@app.get("/api/dataset/stats")
async def dataset_stats():
    """Quick summary stats of the loaded dataset."""
    df = _load_dataset()
    return {
        "total_records":    int(len(df)),
        "columns":          list(df.columns),
        "gender_dist": {
            "male":   int((df["gender"] == 1).sum()),
            "female": int((df["gender"] == 0).sum()),
        },
        "hire_rate": {
            "overall": round(float(df["hiring_decision"].mean()), 3),
            "male":    round(float(df[df["gender"]==1]["hiring_decision"].mean()), 3),
            "female":  round(float(df[df["gender"]==0]["hiring_decision"].mean()), 3),
        },
        "sample_rows": json.loads(df.head(5).to_json(orient="records")),
    }


# Static files last (so API routes are not shadowed)
app.mount("/static", StaticFiles(directory="static"), name="static")
