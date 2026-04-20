"""
Bias Breakers — IndiaHire-Bias Dataset Generator v1.0
Generates a synthetic but research-grounded Indian tech hiring dataset
with documented bias patterns across gender, geography, and institution tier.
"""

import os
import numpy as np
import pandas as pd


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # ── Protected attributes ─────────────────────────────────────────────────
    gender = np.random.choice([0, 1], n, p=[0.43, 0.57])        # 0=Female, 1=Male
    age    = np.random.randint(22, 46, n)
    state  = np.random.choice(
        ["Metro", "Tier2", "Tier3"], n, p=[0.38, 0.37, 0.25]
    )
    institution = np.random.choice(
        ["IIT_NIT", "Tier2_College", "Tier3_College"],
        n, p=[0.18, 0.47, 0.35]
    )

    # ── Legitimate features (should drive outcome) ───────────────────────────
    skills_score     = np.random.normal(68, 14, n).clip(20, 100)
    experience_years = np.random.poisson(4, n).clip(0, 20).astype(int)

    # Career gap — correlated with gender (maternity/care leave patterns)
    gap_years = np.where(
        gender == 0,
        np.random.choice([0, 1, 2, 3], n, p=[0.48, 0.28, 0.15, 0.09]),
        np.random.choice([0, 1, 2, 3], n, p=[0.82, 0.11, 0.05, 0.02]),
    )

    # Referral — network advantage (men receive ~2× more referrals)
    referral = np.where(
        gender == 1,
        np.random.binomial(1, 0.38, n),
        np.random.binomial(1, 0.17, n),
    )

    # Communication score — proxy bias embedded
    comm_score = (
        np.random.normal(65, 12, n)
        + 5 * (gender == 1)
        + 3 * (state == "Metro")
        - 2 * (state == "Tier3")
    ).clip(20, 100)

    # ── Biased hiring decision (log-odds) ────────────────────────────────────
    log_odds = (
        0.04  * skills_score
        + 0.06 * experience_years
        + 0.85 * (gender == 1)                           # male advantage
        - 0.55 * (state == "Tier2")                      # metro preference
        - 1.10 * (state == "Tier3")
        - 0.45 * (institution == "Tier2_College")        # institution prestige
        - 0.85 * (institution == "Tier3_College")
        - 0.35 * gap_years                               # gap penalty
        + 0.55 * referral                                # referral advantage
        + 0.01 * comm_score
        - 4.0                                            # intercept
        + np.random.normal(0, 0.6, n)                    # noise
    )
    prob            = 1.0 / (1.0 + np.exp(-log_odds))
    hiring_decision = np.random.binomial(1, prob, n)

    # ── Actual post-hire performance (NOT correlated with bias factors) ───────
    # This is the smoking gun: rejected candidates perform just as well
    actual_performance = (
        0.75 * skills_score
        + 0.45 * experience_years
        + np.random.normal(0, 8, n)
    ).clip(10, 100).round(1)

    df = pd.DataFrame({
        "gender":                  gender,            # 0=Female, 1=Male
        "age":                     age,
        "state_of_origin":         state,
        "institution_tier":        institution,
        "skills_score":            skills_score.round(1),
        "experience_years":        experience_years,
        "gap_years":               gap_years,
        "referral":                referral,
        "communication_score":     comm_score.round(1),
        "hiring_decision":         hiring_decision,   # TARGET variable
        "actual_performance_score": actual_performance,  # Ground-truth proof
    })

    os.makedirs("datasets", exist_ok=True)
    out = "datasets/indiahire_bias.csv"
    df.to_csv(out, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  IndiaHire-Bias Dataset v1.0  —  Generated Successfully")
    print("=" * 56)
    print(f"  Records          : {n:,}")
    print(f"  Features         : {len(df.columns)}")
    print(f"  Overall hire rate: {df.hiring_decision.mean():.1%}")
    print(f"  Male hire rate   : {df[df.gender==1].hiring_decision.mean():.1%}")
    print(f"  Female hire rate : {df[df.gender==0].hiring_decision.mean():.1%}")
    di = (df[df.gender==0].hiring_decision.mean() /
          df[df.gender==1].hiring_decision.mean())
    print(f"  Disparate Impact : {di:.3f}  (threshold 0.80 -- {'FAIL [X]' if di<0.8 else 'PASS [OK]'})")
    print(f"  Saved to         : {out}")
    print("=" * 56 + "\n")
    return df


if __name__ == "__main__":
    generate()
