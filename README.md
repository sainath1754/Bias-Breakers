# Bias Breakers ⚖️

**AI-Powered Bias Detection, Explanation & Fairness Mitigation Platform**

Bias Breakers detects, explains, and fixes hidden discrimination in AI models — before they affect real people.

> Built for Hackathon — Problem 4: Unbiased AI Decision Making

---

## 🚀 Quick Start

```bash
# 1. Clone & create virtual environment
git clone https://github.com/sainath1754/Bias-Breakers.git
cd Bias-Breakers
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set LLM API key for AI-generated plain-English reports
export ANTHROPIC_API_KEY=your_key_here

# 4. Generate dataset
python generate_dataset.py

# 5. Start server
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## 🎯 What It Does

| Step | Action | Technology |
|------|--------|------------|
| **Detect** | Computes 4 fairness metrics (Disparate Impact, SPD, EOD, AOD) | scikit-learn, pandas |
| **Explain** | SHAP-based mean absolute attribution — pinpoints exact cause of bias | SHAP LinearExplainer |
| **Fix** | Reweighing mitigation — one click, bias reduced before deployment | Custom reweighing algorithm |
| **Report** | Plain-English AI-generated audit report for non-technical stakeholders | Anthropic Claude API |

---

## 📊 Dataset — IndiaHire-Bias v1.0

5,000-record synthetic Indian tech hiring dataset with 6 documented bias patterns:

- **Gender bias** — Women 34% less likely hired despite equal skills
- **Geographic bias** — Tier-3 city candidates penalised 28%
- **Institution bias** — Non-IIT/NIT candidates penalised 41%
- **Career gap penalty** — Disproportionately affects women (maternity/care leave)
- **Referral advantage** — Men receive 2× more referrals
- **Age bias** — Candidates 38+ face 22% lower selection rate

**Ground truth column** (`actual_performance_score`) proves bias was irrational — rejected candidates perform just as well post-hire.

---

## 🛠️ Tech Stack

- **Backend** — Python 3.11, FastAPI, scikit-learn, pandas, numpy
- **Frontend** — HTML5, CSS3, Vanilla JS, Chart.js
- **ML** — Logistic Regression with Reweighing mitigation (IBM AIF360 method)
- **Explainability** — SHAP LinearExplainer (mean |SHAP| attribution per feature)
- **AI Report** — Anthropic Claude API (plain-English audit summary for HR teams)

---

## 📁 Project Structure

```
bias-breakers/
├── main.py               FastAPI application (v3.0)
├── bias_engine.py        ML bias detection + SHAP + LLM report
├── generate_dataset.py   IndiaHire-Bias dataset generator
├── requirements.txt      Dependencies
├── static/index.html     Web frontend
└── datasets/             Generated CSV files
```

---

## 🔗 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web application |
| `/api/demo` | GET | Full audit on IndiaHire-Bias dataset |
| `/api/analyze` | POST | Audit your own CSV file |
| `/api/predict` | POST | Predict outcome for one candidate |
| `/api/compare` | POST | Side-by-side comparison of two profiles |
| `/api/batch` | POST | Batch predict up to 50 candidates |
| `/api/report` | POST | Generate LLM plain-English audit report |
| `/api/dataset/stats` | GET | Dataset summary statistics |
| `/api/health` | GET | Health check |

---

## 🔒 Privacy & Security

- **Uploaded data is processed entirely in-memory** and is never written to disk or stored after the request completes.
- No user data is logged, retained, or shared with third parties.
- All API responses include an `X-Data-Privacy` header confirming this policy.
- The `ANTHROPIC_API_KEY` is read from environment variables only — never hardcoded.

---

## 📈 Scalability

- **Stateless engine** — `BiasEngine` holds no request-level state, making it safe for multi-worker deployment.
- Run with multiple workers: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`
- Large CSVs are streamed into pandas in-memory; no temp files are created.
- Batch endpoint supports up to 50 candidates per call; for larger audits, use `/api/analyze` with a full CSV.

---

## 🎬 Demo Video

> 📹 [Watch the demo](#) — *link to be added*

The demo covers:
1. Running the built-in IndiaHire-Bias audit
2. Scenario comparison — identical profiles, different gender
3. Uploading a custom CSV and getting a fairness report
4. Generating an LLM-written plain-English audit summary

---

## 📎 Links

- 🌐 **Live App**: https://bias-breakers.onrender.com
- 💻 **GitHub**: https://github.com/sainath1754/Bias-Breakers
- 📊 **Project Deck**: *[add link]*

---

## 🏆 Evaluation Alignment

| Criterion | Implementation |
|-----------|---------------|
| Technical complexity | Real train/test split, 5-fold CV, SHAP explanations, reweighing algorithm |
| AI integration | SHAP LinearExplainer + Anthropic Claude API for human-readable reports |
| Performance & scalability | Stateless FastAPI, multi-worker ready, in-memory CSV processing |
| Security & privacy | In-memory only, env-var secrets, privacy response headers |
| Alignment with cause | Detect + Explain + Fix pipeline; covers hiring bias with ground-truth proof |
| Innovation | India-specific dataset with 6 bias patterns; side-by-side scenario comparison; LLM audit reports |