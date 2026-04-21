# Bias Breakers v2.0 ⚖️

**AI-Powered Bias Detection, Explanation & Mitigation Platform**

Bias Breakers detects, explains, and fixes hidden discrimination in AI hiring models — before they affect real people. It uses real ML pipelines with genuine train/test evaluation to prove bias exists and then removes it using Reweighing mitigation.

---

<div align="center">

### 🌐 Live Demo

<a href="https://bias-breakers.onrender.com/">
  <img src="https://img.shields.io/badge/🚀_View_My_Project-Live_on_Render-4c1d95?style=for-the-badge&logo=render&logoColor=white&labelColor=7c3aed" alt="View My Project" height="50"/>
</a>

<br/><br/>

> **✨ Try it now → [https://bias-breakers.onrender.com/](https://bias-breakers.onrender.com/) ✨**
>
> _Experience AI bias detection, explanation & mitigation — live in your browser!_

</div>

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python generate_dataset.py

# 4. Start server
python -m uvicorn main:app --reload

# 5. Open browser
# http://127.0.0.1:8000
```

---

## 🎯 What It Does

| Step | Action |
|------|--------|
| **Detect** | Computes fairness metrics (Disparate Impact, Statistical Parity Difference) on real train/test splits |
| **Explain** | Feature attribution pinpoints the exact cause of bias with contribution percentages |
| **Fix** | Reweighing mitigation adjusts sample weights — one click, bias removed before deployment |
| **Compare** | Side-by-side Bias Exposer shows how two identical profiles get different outcomes due to bias |
| **Predict** | Individual student predictor with rejection reasons, bias factors, and improvement tips |

---

## 📊 Dataset — IndiaHire-Bias v1.0

5,000-record synthetic Indian tech hiring dataset with 6 documented bias patterns:

- **Gender bias** — Women 34% less likely hired despite equal skills
- **Geographic bias** — Tier-3 city candidates penalised 28%
- **Institution bias** — Non-IIT/NIT candidates penalised 41%
- **Career gap penalty** — Disproportionately affects women (maternity/care leave)
- **Referral advantage** — Men receive 2x more referrals
- **Age bias** — Candidates 38+ face 22% lower selection rate

**Ground truth column** (`actual_performance_score`) proves bias was irrational — rejected candidates performed just as well.

---

## 🛠️ Tech Stack

- **Backend** — Python 3.12, FastAPI, Pydantic
- **ML Engine** — scikit-learn (Logistic Regression, Reweighing, Cross-Validation)
- **Data** — pandas, numpy
- **Frontend** — HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Architecture** — REST API, Single-Page Application

---

## 📁 Project Structure

```
bias-breakers/
├── main.py                 FastAPI application (7 API endpoints)
├── bias_engine.py          ML bias detection & mitigation engine
├── generate_dataset.py     Synthetic dataset generator
├── requirements.txt        Python dependencies
├── commands.txt            Setup & run guide
├── README.md               This file
├── static/
│   └── index.html          Frontend web application
└── datasets/
    └── indiahire_bias.csv  Generated dataset (after running generator)
```

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web application |
| `/api/health` | GET | Health check with version info |
| `/api/demo` | GET | Run full audit on IndiaHire dataset |
| `/api/predict` | POST | Predict outcome for a single student |
| `/api/compare` | POST | Compare two profiles side-by-side |
| `/api/batch` | POST | Batch prediction for multiple students |
| `/api/analyze` | POST | Upload and audit your own CSV dataset |
| `/api/dataset/stats` | GET | Quick summary stats of loaded dataset |

---

## 🔬 Key Features

### 1. Bias Exposer (Scenario Comparison)
Compare two profiles — same skills, different demographics. Watch how bias changes the outcome in real time.

### 2. Real ML Metrics
Genuine train/test split (75/25) with actual Accuracy, Precision, Recall, F1 Score, and 5-Fold Cross-Validation.

### 3. Student Predictor
Enter any student's details → get biased vs fair predictions, rejection reasons explained in plain English, bias factors highlighted, and actionable improvement tips.

### 4. One-Click Mitigation
Reweighing algorithm adjusts training sample weights so underrepresented groups carry equal influence. Bias removed with minimal accuracy tradeoff.

### 5. Ground Truth Validation
The `actual_performance_score` column proves the bias was irrational — rejected women would have performed equally or better than those hired.

---

## 📈 ML Pipeline

```
CSV Dataset → Encode & Scale → Train/Test Split (75/25)
    ↓                              ↓
Biased Model (LR)           Reweighed Model (LR + sample weights)
    ↓                              ↓
Fairness Metrics             Fairness Metrics
    ↓                              ↓
    └──── Compare Before vs After ────┘
                   ↓
          Visual Report + Verdicts
```

---

*Built for Hackathon — Problem 4: Unbiased AI Decision Making | v2.0*
