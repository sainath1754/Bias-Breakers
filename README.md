# Bias Breakers ⚖️

**AI-Powered Bias Detection & Fairness Auditing Platform**

Bias Breakers detects, explains, and fixes hidden discrimination in AI models — before they affect real people.

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python generate_dataset.py

# 4. Start server
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

## 🎯 What It Does

| Step | Action |
|------|--------|
| **Detect** | Computes 4 fairness metrics (Disparate Impact, SPD, EOD, AOD) |
| **Explain** | SHAP-based feature attribution — pinpoints the exact cause of bias |
| **Fix** | Reweighing mitigation — one click, bias removed before deployment |

## 📊 Dataset — IndiaHire-Bias v1.0

5,000-record synthetic Indian tech hiring dataset with 6 documented bias patterns:

- **Gender bias** — Women 34% less likely hired despite equal skills
- **Geographic bias** — Tier-3 city candidates penalised 28%
- **Institution bias** — Non-IIT/NIT candidates penalised 41%
- **Career gap penalty** — Disproportionately affects women (maternity)
- **Referral advantage** — Men receive 2× more referrals
- **Age bias** — Candidates 38+ face 22% lower selection rate

**Ground truth column** (`actual_performance_score`) proves bias was irrational.

## 🛠️ Tech Stack

- **Backend** — Python 3.11, FastAPI, scikit-learn, pandas, numpy
- **Frontend** — HTML5, CSS3, Vanilla JS, Chart.js
- **ML** — Logistic Regression, Reweighing mitigation

## 📁 Project Structure

```
bias-breakers/
├── main.py               FastAPI application
├── bias_engine.py        ML bias detection logic
├── generate_dataset.py   Dataset generator
├── requirements.txt      Dependencies
├── static/index.html     Web frontend
└── datasets/             Generated CSV files
```

## 🔗 API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web application |
| `/api/demo` | GET | Run audit on IndiaHire dataset |
| `/api/analyze` | POST | Audit your own CSV file |
| `/api/health` | GET | Health check |

---

*Built for Hackathon — Problem 4: Unbiased AI Decision Making*
