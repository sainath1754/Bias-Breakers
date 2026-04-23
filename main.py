"""
Bias Breakers — FastAPI Backend v3.0

Run: uvicorn main:app --reload

Scalability note:
  This app is fully stateless — BiasEngine holds no request-level state.
  Deploy behind any load balancer with N workers (gunicorn -w 4 uvicorn.workers.UvicornWorker).
  Large CSVs are streamed into pandas in-memory; no temp files are created.
  Uploaded data is NEVER written to disk or stored beyond the request lifetime.
"""

import io
import os
import json
import pathlib
import subprocess
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bias_engine import BiasEngine, SHAP_AVAILABLE

# Auto-generate dataset on cold start (needed on Render.com)
if not pathlib.Path("datasets").exists():
    subprocess.run(["python", "generate_dataset.py"], check=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bias Breakers",
    version="3.0.0",
    description=(
        "AI-Powered Bias Detection, Explanation & Fairness Mitigation Platform. "
        "Uploaded data is processed in-memory and never stored."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

engine = BiasEngine()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dataset() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "datasets", "indiahire_bias.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Dataset missing. Run: python generate_dataset.py"
        )
    return pd.read_csv(path)

def _privacy_headers(response):
    """Add privacy & security headers to every response."""
    response.headers["X-Data-Privacy"]  = "Uploaded data processed in-memory only — never stored."
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

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
        "status":       "ok",
        "project":      "Bias Breakers",
        "version":      "3.0.0",
        "timestamp":    datetime.now().isoformat(),
        "engine":       "BiasEngine v3.0 — Real ML + SHAP Pipeline",
        "shap_enabled": SHAP_AVAILABLE,
        "llm_enabled":  bool(os.environ.get("ANTHROPIC_API_KEY")),
        "data_privacy": "All uploaded data processed in-memory only — never stored.",
        "scalability":  "Stateless workers — horizontal scaling supported.",
    }


@app.get("/api/demo")
async def run_demo():
    """Run full bias audit on the built-in IndiaHire-Bias dataset."""
    df     = _load_dataset()
    result = engine.full_analysis(df)
    return result


@app.post("/api/predict")
async def predict_student(student: dict):
    """
    Predict hiring outcome for a single candidate.
    Returns: biased decision, fair decision, bias factors, improvement tips.
    """
    df = _load_dataset()
    return engine.predict_student(student, df)


@app.post("/api/compare")
async def compare_scenarios(payload: dict = Body(...)):
    """
    Compare two candidate profiles side-by-side to expose bias.
    Body: { "profile_a": {...}, "profile_b": {...} }
    """
    profile_a = payload.get("profile_a")
    profile_b = payload.get("profile_b")
    if not profile_a or not profile_b:
        raise HTTPException(status_code=422, detail="Both profile_a and profile_b are required.")
    df = _load_dataset()
    return engine.compare_scenarios(profile_a, profile_b, df)


@app.post("/api/batch")
async def batch_predict(payload: dict = Body(...)):
    """
    Predict outcomes for up to 50 candidates at once.
    Body: { "students": [{...}, ...] }
    Scalability: for larger batches, split into multiple requests or use /api/analyze.
    """
    students = payload.get("students", [])
    if not students:
        raise HTTPException(status_code=422, detail="No students provided.")
    if len(students) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 candidates per batch.")
    df = _load_dataset()
    return engine.batch_predict(students, df)


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """
    Audit a user-uploaded CSV for bias.
    PRIVACY: data is processed entirely in-memory and never written to disk or stored.
    Required columns: gender (0=Female, 1=Male), hiring_decision (0/1)
    """
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
                detail=f"Missing required column '{col}'. "
                       "CSV must contain: gender (0=Female/1=Male), hiring_decision (0/1)."
            )

    result = engine.full_analysis(df)
    result["privacy_notice"] = "This data was processed in-memory only and has not been stored."
    return result


@app.post("/api/report")
async def generate_report(payload: dict = Body(...)):
    """
    Generate a plain-English LLM-powered bias audit report.
    Body: the full JSON object returned by /api/demo or /api/analyze.
    Requires ANTHROPIC_API_KEY environment variable to be set.
    Falls back to a template report if the API key is absent.
    """
    if not payload:
        raise HTTPException(status_code=422, detail="Pass the full analysis result as the request body.")
    report = BiasEngine.generate_llm_report(payload)
    return {
        "report":    report,
        "llm_used":  bool(os.environ.get("ANTHROPIC_API_KEY")),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/dataset/stats")
async def dataset_stats():
    """Quick summary statistics of the built-in IndiaHire-Bias dataset."""
    df = _load_dataset()
    return {
        "total_records": int(len(df)),
        "columns":       list(df.columns),
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


# Static files (must be last so API routes are not shadowed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Render.com / local entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)