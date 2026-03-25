"""
==================================================================================
 03_ml_pipeline.py — Full ML Pipeline for Churn Prediction
==================================================================================
 Project  : Customer Churn Analysis for Telecom Industry
 Intern   : Data Analyst Internship — Elevate Labs

 Pipeline Steps:
   1. Data Loading & Cleaning
   2. Encoding Categorical Variables
   3. Feature Scaling
   4. Exploratory Data Analysis (EDA) with visualizations
   5. Class Imbalance Handling (SMOTE)
   6. Train/Test Split
   7. Model Training (Logistic Regression + Random Forest)
   8. Model Comparison (Accuracy, Precision, Recall, F1, AUC)
   9. Confusion Matrix visualization
  10. Feature Importance (Random Forest)
  11. SHAP Explainability
  12. Customer Segmentation (At Risk / Dormant / Loyal)
==================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving charts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "data", "telecom_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR  = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(CHART_DIR, exist_ok=True)

# ─── Plotting Style ─────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "primary"  : "#2563EB",
    "secondary": "#F59E0B",
    "success"  : "#10B981",
    "danger"   : "#EF4444",
    "info"     : "#8B5CF6",
    "bg"       : "#0F172A",
    "text"     : "#E2E8F0",
    "churn"    : ["#10B981", "#EF4444"],  # Active, Churned
}


def save_chart(fig, filename, dpi=150):
    """Helper to save charts consistently."""
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    💾 Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def load_and_clean(path):
    """
    Load the dataset and perform data cleaning:
      - Handle missing values
      - Convert data types
      - Remove duplicates

    Business Rationale:
      Real telecom data is messy — billing systems produce blanks in
      TotalCharges, logging gaps create NaN in CallDuration. Handling
      these properly prevents model training failures and bias.
    """
    print("\n" + "=" * 60)
    print(" 📦 STEP 1: Data Loading & Cleaning")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ── Missing values summary ───────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n  Missing values found:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"    {col}: {count} ({pct:.1f}%)")

    # ── Handle missing TotalCharges ──────────────────────────────────────
    # Business logic: Impute with MonthlyCharges × Tenure (expected value)
    if df["TotalCharges"].isnull().any():
        mask = df["TotalCharges"].isnull()
        df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "Tenure"]
        print(f"  ✅ Imputed {mask.sum()} TotalCharges using MonthlyCharges × Tenure")

    # ── Handle missing CallDuration ──────────────────────────────────────
    # Business logic: Use median (robust to outliers)
    if df["CallDuration"].isnull().any():
        median_cd = df["CallDuration"].median()
        count_cd = df["CallDuration"].isnull().sum()
        df["CallDuration"].fillna(median_cd, inplace=True)
        print(f"  ✅ Imputed {count_cd} CallDuration with median ({median_cd:.1f})")

    # ── Remove duplicates ────────────────────────────────────────────────
    dups = df.duplicated().sum()
    if dups > 0:
        df.drop_duplicates(inplace=True)
        print(f"  ✅ Removed {dups} duplicate rows")
    else:
        print(f"  ✅ No duplicate rows found")

    # ── Data types ───────────────────────────────────────────────────────
    print(f"\n  Data types:")
    print(df.dtypes.to_string())

    print(f"\n  Final shape after cleaning: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ENCODING CATEGORICAL VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
def encode_features(df):
    """
    Encode categorical columns using Label Encoding.

    Business Rationale:
      ML models require numeric input. Label encoding is used here for
      ordinal-like categories (ContractType: Month-to-Month < One Year < Two Year).
      For nominal categories (Gender, PaymentMethod), one-hot encoding is an
      alternative, but Label Encoding keeps the feature space small.
    """
    print("\n" + "=" * 60)
    print(" 🔤 STEP 2: Encoding Categorical Variables")
    print("=" * 60)

    df_encoded = df.copy()
    label_encoders = {}

    categorical_cols = df_encoded.select_dtypes(include=["object"]).columns
    categorical_cols = [c for c in categorical_cols if c != "CustomerID"]

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  ✅ Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df_encoded, label_encoders


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════════════
def scale_features(X_train, X_test):
    """
    Standard scaling (z-score normalization) for numeric features.

    Business Rationale:
      Logistic Regression is sensitive to feature scales — MonthlyCharges
      (18-120) would dominate Tenure (1-72) without scaling. Random Forest
      is scale-invariant, but we scale for consistency across models.
      Always fit on training data ONLY to prevent data leakage.
    """
    print("\n" + "=" * 60)
    print(" 📏 STEP 3: Feature Scaling")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)     # transform only — no fit!

    print(f"  ✅ StandardScaler fitted on {X_train.shape[0]} training samples")
    print(f"  ✅ Transformed {X_test.shape[0]} test samples")

    return X_train_scaled, X_test_scaled, scaler


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════════════
def run_eda(df, df_encoded):
    """
    Generate exploratory visualizations to understand churn patterns.

    Business Rationale:
      EDA reveals patterns BEFORE modeling. These charts answer:
      - How imbalanced is the churn label?
      - Which features differ most between churned/active customers?
      - What correlations exist between features?
    """
    print("\n" + "=" * 60)
    print(" 📊 STEP 4: Exploratory Data Analysis")
    print("=" * 60)

    # ── 4.1 Churn Distribution ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    counts = df["Churn"].value_counts()
    labels = ["Active (0)", "Churned (1)"]
    bars = ax.bar(labels, counts.values, color=COLORS["churn"],
                  edgecolor="white", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                fontweight="bold", fontsize=12, color=COLORS["text"])

    ax.set_title("Churn Distribution", fontsize=16, fontweight="bold",
                 color=COLORS["text"], pad=15)
    ax.set_ylabel("Number of Customers", fontsize=12, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "01_churn_distribution.png")

    # ── 4.2 Tenure vs Churn (KDE Plot) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for churn_val, label, color in [(0, "Active", COLORS["success"]),
                                      (1, "Churned", COLORS["danger"])]:
        subset = df[df["Churn"] == churn_val]["Tenure"]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color,
                edgecolor="white", linewidth=0.5)

    ax.set_title("Tenure Distribution by Churn Status", fontsize=16,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_xlabel("Tenure (Months)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Frequency", fontsize=12, color=COLORS["text"])
    ax.legend(fontsize=11, facecolor=COLORS["bg"], edgecolor="#334155",
              labelcolor=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "02_tenure_vs_churn.png")

    # ── 4.3 Monthly Charges vs Churn (Box Plot) ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    data_active  = df[df["Churn"] == 0]["MonthlyCharges"]
    data_churned = df[df["Churn"] == 1]["MonthlyCharges"]

    bp = ax.boxplot([data_active, data_churned],
                    labels=["Active", "Churned"],
                    patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=COLORS["text"]),
                    capprops=dict(color=COLORS["text"]),
                    flierprops=dict(markerfacecolor=COLORS["secondary"],
                                   markersize=3, alpha=0.5))

    for patch, color in zip(bp["boxes"], COLORS["churn"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Monthly Charges by Churn Status", fontsize=16,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_ylabel("Monthly Charges ($)", fontsize=12, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "03_monthly_charges_vs_churn.png")

    # ── 4.4 Correlation Heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # Use encoded dataframe for correlation (numeric only)
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    corr_matrix = df_encoded[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax,
                annot_kws={"size": 8, "color": "white"},
                linewidths=0.5, linecolor="#1E293B",
                cbar_kws={"shrink": 0.8})

    ax.set_title("Feature Correlation Heatmap", fontsize=16,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_chart(fig, "04_correlation_heatmap.png")

    # ── 4.5 Contract Type vs Churn ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    ct_churn = df.groupby("ContractType")["Churn"].mean() * 100
    ct_churn = ct_churn.sort_values(ascending=False)

    bars = ax.barh(ct_churn.index, ct_churn.values,
                   color=[COLORS["danger"], COLORS["secondary"], COLORS["success"]],
                   edgecolor="white", linewidth=1, height=0.4)

    for bar, val in zip(bars, ct_churn.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontweight="bold",
                fontsize=12, color=COLORS["text"])

    ax.set_title("Churn Rate by Contract Type", fontsize=16,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_xlabel("Churn Rate (%)", fontsize=12, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "05_contract_type_churn.png")

    # ── 4.6 Complaints vs Churn ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    comp_churn = df.groupby("Complaints")["Churn"].mean() * 100
    ax.bar(comp_churn.index, comp_churn.values, color=COLORS["info"],
           edgecolor="white", linewidth=0.8)

    ax.set_title("Churn Rate by Number of Complaints", fontsize=16,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_xlabel("Number of Complaints", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Churn Rate (%)", fontsize=12, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "06_complaints_vs_churn.png")

    print("  ✅ Generated 6 EDA charts")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CLASS IMBALANCE HANDLING (SMOTE)
# ══════════════════════════════════════════════════════════════════════════════
def handle_imbalance(X_train, y_train):
    """
    Apply SMOTE (Synthetic Minority Oversampling) to balance classes.

    Business Rationale:
      Telecom churn is typically 20-30% — an imbalanced problem. Without
      SMOTE, models optimize for accuracy by predicting "Active" for
      everyone (80% accuracy but useless for identifying churners).
      SMOTE creates synthetic churned samples so the model learns the
      churn boundary better.
    """
    print("\n" + "=" * 60)
    print(" ⚖️  STEP 5: Handling Class Imbalance (SMOTE)")
    print("=" * 60)

    print(f"  Before SMOTE:")
    print(f"    Active:  {(y_train == 0).sum()}")
    print(f"    Churned: {(y_train == 1).sum()}")
    print(f"    Ratio:   {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")

    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"\n  After SMOTE:")
        print(f"    Active:  {(y_resampled == 0).sum()}")
        print(f"    Churned: {(y_resampled == 1).sum()}")
        print(f"    Ratio:   {(y_resampled == 0).sum() / (y_resampled == 1).sum():.2f}:1")
        print(f"  ✅ SMOTE applied successfully")

        return X_resampled, y_resampled

    except ImportError:
        print("  ⚠️  imbalanced-learn not installed. Skipping SMOTE.")
        print("     Install with: pip install imbalanced-learn")
        return X_train, y_train


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 & 7: TRAIN/TEST SPLIT + MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Logistic Regression and Random Forest classifiers.

    Business Rationale:
      - Logistic Regression: Interpretable, good baseline. C-suite wants
        to know "what increases churn probability by X%?" — LR coefficients
        directly answer this.
      - Random Forest: Higher accuracy, handles non-linear relationships.
        Better for production scoring but less interpretable.

      We compare both to demonstrate trade-off between
      interpretability vs performance.
    """
    print("\n" + "=" * 60)
    print(" 🤖 STEP 6 & 7: Train/Test Split + Model Training")
    print("=" * 60)

    results = {}

    # ── Model 1: Logistic Regression ─────────────────────────────────────
    print("\n  📌 Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "model"     : lr_model,
        "y_pred"    : lr_pred,
        "y_proba"   : lr_proba,
        "accuracy"  : accuracy_score(y_test, lr_pred),
        "precision" : precision_score(y_test, lr_pred),
        "recall"    : recall_score(y_test, lr_pred),
        "f1"        : f1_score(y_test, lr_pred),
        "auc"       : roc_auc_score(y_test, lr_proba),
    }
    print(f"    Accuracy:  {results['Logistic Regression']['accuracy']:.4f}")
    print(f"    AUC:       {results['Logistic Regression']['auc']:.4f}")

    # ── Model 2: Random Forest ───────────────────────────────────────────
    print("\n  📌 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    results["Random Forest"] = {
        "model"     : rf_model,
        "y_pred"    : rf_pred,
        "y_proba"   : rf_proba,
        "accuracy"  : accuracy_score(y_test, rf_pred),
        "precision" : precision_score(y_test, rf_pred),
        "recall"    : recall_score(y_test, rf_pred),
        "f1"        : f1_score(y_test, rf_pred),
        "auc"       : roc_auc_score(y_test, rf_proba),
    }
    print(f"    Accuracy:  {results['Random Forest']['accuracy']:.4f}")
    print(f"    AUC:       {results['Random Forest']['auc']:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def compare_models(results, y_test):
    """
    Compare all models using standard classification metrics.

    Business Rationale:
      For churn prediction, RECALL is more important than precision.
      Missing a churner (false negative) costs real revenue. A false
      alarm (false positive) only wastes a retention call — much cheaper.
    """
    print("\n" + "=" * 60)
    print(" 📊 STEP 8: Model Comparison")
    print("=" * 60)

    # ── Metrics Table ────────────────────────────────────────────────────
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            "Model"     : name,
            "Accuracy"  : round(res["accuracy"], 4),
            "Precision" : round(res["precision"], 4),
            "Recall"    : round(res["recall"], 4),
            "F1 Score"  : round(res["f1"], 4),
            "AUC-ROC"   : round(res["auc"], 4),
        })

    metrics_df = pd.DataFrame(metrics_data)
    print("\n" + metrics_df.to_string(index=False))

    # Save to CSV
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  ✅ Saved to {metrics_path}")

    # ── Model Comparison Bar Chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    x = np.arange(len(metric_names))
    width = 0.30
    model_colors = [COLORS["primary"], COLORS["secondary"]]

    for i, (name, res) in enumerate(results.items()):
        values = [res["accuracy"], res["precision"], res["recall"],
                  res["f1"], res["auc"]]
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=model_colors[i], edgecolor="white",
                      linewidth=0.8, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=COLORS["text"])

    ax.set_xlabel("Metrics", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Score", fontsize=12, color=COLORS["text"])
    ax.set_title("Model Performance Comparison", fontsize=16,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, facecolor=COLORS["bg"], edgecolor="#334155",
              labelcolor=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "07_model_comparison.png")

    # ── ROC Curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for (name, res), color in zip(results.items(), model_colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})",
                color=color, linewidth=2.5)

    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)
    ax.set_title("ROC Curve Comparison", fontsize=16, fontweight="bold",
                 color=COLORS["text"], pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("True Positive Rate", fontsize=12, color=COLORS["text"])
    ax.legend(fontsize=11, facecolor=COLORS["bg"], edgecolor="#334155",
              labelcolor=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "08_roc_curve.png")

    return metrics_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(results, y_test):
    """
    Plot confusion matrices for all models.

    Business Rationale:
      The confusion matrix breaks down:
      - True Positives: Correctly identified churners → saved by retention
      - False Negatives: Missed churners → revenue lost
      - False Positives: Unnecessary retention calls → small cost
      - True Negatives: Correctly identified active → no action needed
    """
    print("\n" + "=" * 60)
    print(" 🎯 STEP 9: Confusion Matrix")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS["bg"])

    for ax, (name, res) in zip(axes, results.items()):
        ax.set_facecolor(COLORS["bg"])
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Active", "Churned"],
                    yticklabels=["Active", "Churned"],
                    annot_kws={"size": 14, "fontweight": "bold"},
                    linewidths=2, linecolor=COLORS["bg"])
        ax.set_title(f"{name}", fontsize=14, fontweight="bold",
                     color=COLORS["text"], pad=10)
        ax.set_xlabel("Predicted", fontsize=11, color=COLORS["text"])
        ax.set_ylabel("Actual", fontsize=11, color=COLORS["text"])
        ax.tick_params(colors=COLORS["text"], labelsize=10)

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold",
                 color=COLORS["text"], y=1.02)
    save_chart(fig, "09_confusion_matrix.png")

    # Print classification reports
    for name, res in results.items():
        print(f"\n  📋 {name} Classification Report:")
        print(classification_report(y_test, res["y_pred"],
                                    target_names=["Active", "Churned"]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(rf_model, feature_names):
    """
    Plot Random Forest feature importance.

    Business Rationale:
      Feature importance tells the business WHERE to invest:
      - If ContractType is top → invest in contract conversion campaigns
      - If Complaints is top → invest in customer service quality
      - If Tenure is top → invest in onboarding & early engagement
    """
    print("\n" + "=" * 60)
    print(" 🏆 STEP 10: Feature Importance (Random Forest)")
    print("=" * 60)

    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature"    : feature_names,
        "Importance" : importances
    }).sort_values("Importance", ascending=True)

    print("\n  Feature Importances:")
    for _, row in fi_df.iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"    {row['Feature']:25s} {row['Importance']:.4f}  {bar}")

    # ── Chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors,
            edgecolor="white", linewidth=0.5, height=0.6)

    for i, (imp, feat) in enumerate(zip(fi_df["Importance"], fi_df["Feature"])):
        ax.text(imp + 0.003, i, f"{imp:.4f}", va="center",
                fontsize=10, fontweight="bold", color=COLORS["text"])

    ax.set_title("Random Forest — Feature Importance", fontsize=16,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_xlabel("Importance", fontsize=12, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    save_chart(fig, "10_feature_importance.png")

    return fi_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
def run_shap_analysis(rf_model, X_test, feature_names):
    """
    SHAP (SHapley Additive exPlanations) analysis for model explainability.

    Business Rationale:
      SHAP answers "WHY did the model predict this customer will churn?"
      for each individual customer — critical for:
      - Regulatory compliance (model transparency)
      - Targeted retention (personalized offers)
      - Trust-building with business stakeholders
    """
    print("\n" + "=" * 60)
    print(" 🔍 STEP 11: SHAP Explainability")
    print("=" * 60)

    try:
        import shap

        # Use TreeExplainer for Random Forest (fast, exact)
        explainer = shap.TreeExplainer(rf_model)

        # Sample for speed (SHAP on full test set can be slow)
        sample_size = min(500, X_test.shape[0])
        X_sample = X_test[:sample_size]

        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values is a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Class 1 = Churned
        else:
            shap_vals = shap_values

        # ── SHAP Summary Plot (Beeswarm) ────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
        shap.summary_plot(shap_vals, X_sample,
                          feature_names=feature_names,
                          show=False, max_display=13)
        plt.title("SHAP Summary — Impact on Churn Prediction",
                  fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        save_chart(plt.gcf(), "11_shap_summary.png", dpi=150)

        # ── SHAP Bar Plot (Mean Absolute SHAP) ──────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        shap.summary_plot(shap_vals, X_sample,
                          feature_names=feature_names,
                          plot_type="bar", show=False, max_display=13)
        plt.title("SHAP — Mean Feature Impact on Churn",
                  fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        save_chart(plt.gcf(), "12_shap_bar.png", dpi=150)

        print("  ✅ SHAP analysis complete — 2 charts saved")

    except ImportError:
        print("  ⚠️  SHAP not installed. Skipping SHAP analysis.")
        print("     Install with: pip install shap")
    except Exception as e:
        print(f"  ⚠️  SHAP analysis failed: {e}")
        print("     This is non-critical — other analyses are unaffected.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def segment_customers(df, rf_model, X_full_scaled, feature_names):
    """
    Segment customers by churn probability into actionable groups.

    Business Rationale:
      Not all customers need the same retention strategy:
      - AT RISK (>50% churn prob): Immediate intervention needed
      - DORMANT (20-50%): Proactive engagement to prevent deterioration
      - LOYAL (<20%): Reward & upsell opportunities

      Each segment gets different budget allocation and communication.
    """
    print("\n" + "=" * 60)
    print(" 👥 STEP 12: Customer Segmentation")
    print("=" * 60)

    # Get churn probabilities for ALL customers
    churn_probabilities = rf_model.predict_proba(X_full_scaled)[:, 1]

    # Assign segments
    segments = []
    for prob in churn_probabilities:
        if prob >= 0.50:
            segments.append("At Risk")
        elif prob >= 0.20:
            segments.append("Dormant")
        else:
            segments.append("Loyal")

    df_segmented = df.copy()
    df_segmented["ChurnProbability"] = np.round(churn_probabilities, 4)
    df_segmented["Segment"]          = segments

    # ── Segment Summary ──────────────────────────────────────────────────
    seg_summary = df_segmented.groupby("Segment").agg(
        CustomerCount    = ("CustomerID", "count"),
        AvgChurnProb     = ("ChurnProbability", "mean"),
        AvgMonthlyCharge = ("MonthlyCharges", "mean"),
        AvgTenure        = ("Tenure", "mean"),
        AvgComplaints    = ("Complaints", "mean"),
        ActualChurnRate  = ("Churn", "mean"),
    ).round(4)

    print("\n  Segment Summary:")
    print(seg_summary.to_string())

    # ── Save segment data ────────────────────────────────────────────────
    seg_path = os.path.join(OUTPUT_DIR, "customer_segments.csv")
    seg_summary.to_csv(seg_path)
    print(f"\n  ✅ Segment summary saved to {seg_path}")

    full_path = os.path.join(OUTPUT_DIR, "all_customers_segmented.csv")
    df_segmented.to_csv(full_path, index=False)
    print(f"  ✅ Full segmented data saved to {full_path}")

    # ── Segment Visualization — Donut Chart ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS["bg"])

    # Donut chart
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    seg_counts = df_segmented["Segment"].value_counts()
    seg_colors = [COLORS["danger"], COLORS["secondary"], COLORS["success"]]

    wedges, texts, autotexts = ax.pie(
        seg_counts.values, labels=seg_counts.index,
        colors=seg_colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor=COLORS["bg"], linewidth=2)
    )
    for text in texts + autotexts:
        text.set_color(COLORS["text"])
        text.set_fontsize(11)
        text.set_fontweight("bold")
    ax.set_title("Customer Segments", fontsize=14, fontweight="bold",
                 color=COLORS["text"], pad=15)

    # Bar chart — metrics by segment
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])

    seg_order = ["At Risk", "Dormant", "Loyal"]
    avg_probs = [seg_summary.loc[s, "AvgChurnProb"] * 100
                 if s in seg_summary.index else 0 for s in seg_order]

    bars = ax.bar(seg_order, avg_probs, color=seg_colors,
                  edgecolor="white", linewidth=1, width=0.5)

    for bar, val in zip(bars, avg_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontweight="bold",
                fontsize=12, color=COLORS["text"])

    ax.set_title("Average Churn Probability by Segment", fontsize=14,
                 fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_ylabel("Churn Probability (%)", fontsize=12, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    save_chart(fig, "13_customer_segments.png")

    return df_segmented


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    """Master function — orchestrates the entire ML pipeline."""

    print("\n" + "🔷" * 30)
    print("  TELECOM CUSTOMER CHURN — ML PIPELINE")
    print("🔷" * 30)

    # ── Step 1: Load & Clean ─────────────────────────────────────────────
    df = load_and_clean(DATA_PATH)

    # ── Step 2: Encode ───────────────────────────────────────────────────
    df_encoded, label_encoders = encode_features(df)

    # ── Step 3: Prepare features ─────────────────────────────────────────
    feature_cols = [c for c in df_encoded.columns if c not in ["CustomerID", "Churn"]]
    X = df_encoded[feature_cols].values
    y = df_encoded["Churn"].values

    print(f"\n  Feature matrix: {X.shape}")
    print(f"  Target vector:  {y.shape}")
    print(f"  Features: {feature_cols}")

    # ── Step 4: EDA ──────────────────────────────────────────────────────
    run_eda(df, df_encoded)

    # ── Step 5: Train/Test Split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n  Train/Test split: {X_train.shape[0]} / {X_test.shape[0]}")

    # ── Step 6: Scale ────────────────────────────────────────────────────
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ── Step 7: Handle Imbalance ─────────────────────────────────────────
    X_train_balanced, y_train_balanced = handle_imbalance(X_train_scaled, y_train)

    # ── Step 8 & 9: Train Models ─────────────────────────────────────────
    results = train_models(X_train_balanced, X_test_scaled,
                           y_train_balanced, y_test, feature_cols)

    # ── Step 10: Compare ─────────────────────────────────────────────────
    metrics_df = compare_models(results, y_test)

    # ── Step 11: Confusion Matrix ────────────────────────────────────────
    plot_confusion_matrix(results, y_test)

    # ── Step 12: Feature Importance ──────────────────────────────────────
    rf_model = results["Random Forest"]["model"]
    fi_df = plot_feature_importance(rf_model, feature_cols)

    # ── Step 13: SHAP ────────────────────────────────────────────────────
    run_shap_analysis(rf_model, X_test_scaled, feature_cols)

    # ── Step 14: Segmentation ────────────────────────────────────────────
    X_full_scaled = scaler.transform(X)
    df_segmented = segment_customers(df, rf_model, X_full_scaled, feature_cols)

    # ── Final Summary ────────────────────────────────────────────────────
    print("\n" + "🔷" * 30)
    print("  ✅ PIPELINE COMPLETE — All outputs saved to ./outputs/")
    print("🔷" * 30)
    print(f"\n  📊 Charts: {CHART_DIR}")
    print(f"  📋 Metrics: {os.path.join(OUTPUT_DIR, 'model_metrics.csv')}")
    print(f"  👥 Segments: {os.path.join(OUTPUT_DIR, 'customer_segments.csv')}")
    print(f"  📁 Full data: {os.path.join(OUTPUT_DIR, 'all_customers_segmented.csv')}")

    return df_segmented, results, metrics_df


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_segmented, results, metrics_df = main()
