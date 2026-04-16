"""
Bias Breakers — FastAPI Backend
Run: uvicorn main:app --reload
"""

import io
import os

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from bias_engine import BiasEngine

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Bias Breakers", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = BiasEngine()

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
    return {"status": "ok", "project": "Bias Breakers", "version": "1.0.0"}


@app.get("/api/demo")
async def run_demo():
    path = os.path.join(os.path.dirname(__file__), "datasets", "indiahire_bias.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Dataset missing. Run:  python generate_dataset.py"
        )
    df     = pd.read_csv(path)
    result = engine.full_analysis(df)
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


# Static files last (so API routes are not shadowed)
app.mount("/static", StaticFiles(directory="static"), name="static")
