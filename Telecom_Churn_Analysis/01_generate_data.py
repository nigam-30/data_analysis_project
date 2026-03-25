"""
==================================================================================
 01_generate_data.py — Synthetic Telecom Churn Dataset Generator
==================================================================================
 Project  : Customer Churn Analysis for Telecom Industry
 Intern   : Data Analyst Internship — Elevate Labs
 Purpose  : Creates a realistic synthetic dataset of 5,000 telecom customers
            with churn labels correlated to business-meaningful features.

 Business Logic:
   - Month-to-Month contracts have highest churn (~33%)
   - More complaints → higher churn probability
   - Lower tenure → higher churn
   - High monthly charges + no value-adds (OnlineSecurity, TechSupport) → churn
   - Payment delays and low recharge frequency signal disengagement
==================================================================================
"""

import numpy as np
import pandas as pd
import os

# ─── Configuration ───────────────────────────────────────────────────────────
np.random.seed(42)           # Reproducibility
N_CUSTOMERS = 5000           # Total rows to generate
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")

# ─── Helper: Weighted Churn Probability ──────────────────────────────────────
def compute_churn_probability(row):
    """
    Compute a churn probability between 0 and 1 based on multiple
    business-relevant features.  This mimics real-world churn drivers:
      - Contract type (month-to-month = risky)
      - Tenure (new customers churn more)
      - Complaints (service dissatisfaction)
      - Monthly charges vs perceived value
      - Recharge frequency (engagement proxy)
      - Call duration (usage engagement)
    """
    prob = 0.10  # baseline 10% churn

    # Contract type impact
    if row["ContractType"] == "Month-to-Month":
        prob += 0.20
    elif row["ContractType"] == "One Year":
        prob += 0.05

    # Tenure impact — new customers are riskier
    if row["Tenure"] <= 6:
        prob += 0.15
    elif row["Tenure"] <= 12:
        prob += 0.08
    elif row["Tenure"] >= 48:
        prob -= 0.10

    # Complaints impact — each complaint adds risk
    if row["Complaints"] >= 4:
        prob += 0.25
    elif row["Complaints"] >= 2:
        prob += 0.12
    elif row["Complaints"] == 0:
        prob -= 0.05

    # High charges + no online security = poor value perception
    if row["MonthlyCharges"] > 80 and row["OnlineSecurity"] == "No":
        prob += 0.10

    # No tech support amplifies frustration
    if row["TechSupport"] == "No" and row["Complaints"] >= 2:
        prob += 0.08

    # Recharge frequency (low = disengagement)
    if row["RechargeFrequency"] <= 2:
        prob += 0.12
    elif row["RechargeFrequency"] >= 8:
        prob -= 0.05

    # Low call duration = low usage = low value
    if row["CallDuration"] < 100:
        prob += 0.08

    # Payment method — electronic check users churn more (industry pattern)
    if row["PaymentMethod"] == "Electronic Check":
        prob += 0.06

    return np.clip(prob, 0.02, 0.95)


def generate_dataset():
    """Main generation function — builds the full DataFrame."""

    print("=" * 60)
    print(" 📊 Generating Synthetic Telecom Churn Dataset")
    print("=" * 60)

    # ─── Customer IDs ────────────────────────────────────────────────────
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]

    # ─── Demographic Features ────────────────────────────────────────────
    genders        = np.random.choice(["Male", "Female"], N_CUSTOMERS, p=[0.50, 0.50])
    senior_citizen = np.random.choice([0, 1], N_CUSTOMERS, p=[0.84, 0.16])

    # ─── Service Features ────────────────────────────────────────────────
    tenure           = np.random.exponential(scale=24, size=N_CUSTOMERS).astype(int)
    tenure           = np.clip(tenure, 1, 72)  # 1 to 72 months

    monthly_charges  = np.round(np.random.uniform(18.0, 120.0, N_CUSTOMERS), 2)
    total_charges    = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.05, N_CUSTOMERS), 2)

    contract_types   = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        N_CUSTOMERS, p=[0.50, 0.25, 0.25]
    )

    internet_service = np.random.choice(
        ["DSL", "Fiber Optic", "No"], N_CUSTOMERS, p=[0.35, 0.45, 0.20]
    )

    online_security  = np.where(
        internet_service == "No", "No Internet Service",
        np.random.choice(["Yes", "No"], N_CUSTOMERS, p=[0.40, 0.60])
    )

    tech_support     = np.where(
        internet_service == "No", "No Internet Service",
        np.random.choice(["Yes", "No"], N_CUSTOMERS, p=[0.35, 0.65])
    )

    # ─── Payment & Billing ───────────────────────────────────────────────
    payment_methods  = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        N_CUSTOMERS, p=[0.35, 0.20, 0.25, 0.20]
    )

    # ─── Engagement Features ─────────────────────────────────────────────
    call_duration      = np.round(np.random.uniform(30, 600, N_CUSTOMERS), 1)  # minutes/month
    complaints         = np.random.poisson(lam=1.5, size=N_CUSTOMERS)
    complaints         = np.clip(complaints, 0, 8)
    recharge_frequency = np.random.poisson(lam=5, size=N_CUSTOMERS)
    recharge_frequency = np.clip(recharge_frequency, 0, 12)

    # ─── Assemble DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame({
        "CustomerID"       : customer_ids,
        "Gender"           : genders,
        "SeniorCitizen"    : senior_citizen,
        "Tenure"           : tenure,
        "MonthlyCharges"   : monthly_charges,
        "TotalCharges"     : total_charges,
        "ContractType"     : contract_types,
        "InternetService"  : internet_service,
        "OnlineSecurity"   : online_security,
        "TechSupport"      : tech_support,
        "PaymentMethod"    : payment_methods,
        "CallDuration"     : call_duration,
        "Complaints"       : complaints,
        "RechargeFrequency": recharge_frequency,
    })

    # ─── Introduce ~3% Missing Values (realistic) ────────────────────────
    # TotalCharges sometimes has blanks in real telecom data
    mask_tc = np.random.rand(N_CUSTOMERS) < 0.03
    df.loc[mask_tc, "TotalCharges"] = np.nan

    # CallDuration may have holes from logging issues
    mask_cd = np.random.rand(N_CUSTOMERS) < 0.02
    df.loc[mask_cd, "CallDuration"] = np.nan

    print(f"  ✅ Generated {N_CUSTOMERS} customers with 14 raw features")
    print(f"  ✅ Missing values injected: TotalCharges={mask_tc.sum()}, CallDuration={mask_cd.sum()}")

    # ─── Compute Churn Label ─────────────────────────────────────────────
    churn_probs = df.apply(compute_churn_probability, axis=1)
    df["Churn"] = (np.random.rand(N_CUSTOMERS) < churn_probs).astype(int)

    churn_rate = df["Churn"].mean() * 100
    print(f"  ✅ Churn label generated — overall churn rate: {churn_rate:.1f}%")

    # ─── Save ────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "telecom_churn.csv")
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved to {output_path}")
    print(f"  📐 Shape: {df.shape}")
    print("=" * 60)

    return df


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset()
    print("\nFirst 5 rows:\n")
    print(df.head().to_string(index=False))
