# 📊 Customer Churn Analysis — Telecom Industry

> **Data Analyst Internship · Elevate Labs**   
> End-to-End Churn Prediction using SQL + Python + Machine Learning

---

## 🎯 Objective

Build a complete churn prediction system that identifies at-risk telecom customers **before** they leave, enabling proactive retention strategies that can save 15-25% of revenue at risk.

---

## 📁 Project Structure

```
Telecom_Churn_Analysis/
├── 01_generate_data.py          ← Synthetic dataset (5,000 customers)
├── 02_sql_analysis.py           ← 10 SQL queries via SQLite
├── 03_ml_pipeline.py            ← ML + SHAP + Segmentation
├── 04_generate_report.py        ← HTML report generator
├── run_pipeline.py              ← Master runner (all steps)
├── README.md
├── sql/
│   └── churn_queries.sql        ← Standalone SQL file
├── data/
│   └── telecom_churn.csv        ← Generated dataset
└── outputs/
    ├── Telecom_Churn_Analysis_Report.html  ← 📊 MAIN REPORT
    ├── sql_aggregations.xlsx               ← SQL results (10 sheets)
    ├── model_metrics.csv                   ← Model comparison
    ├── customer_segments.csv               ← Segment statistics
    ├── all_customers_segmented.csv         ← Full segmented data
    └── charts/                             ← Publication-quality charts
        ├── 01_churn_distribution.png
        ├── 02_tenure_vs_churn.png
        ├── 03_monthly_charges_vs_churn.png
        ├── 04_correlation_heatmap.png
        ├── 05_contract_type_churn.png
        ├── 06_complaints_vs_churn.png
        ├── 07_model_comparison.png
        ├── 08_roc_curve.png
        ├── 09_confusion_matrix.png
        ├── 10_feature_importance.png
        ├── 11_shap_summary.png
        ├── 12_shap_bar.png
        └── 13_customer_segments.png
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap openpyxl
```

### Full Pipeline (Recommended)
```bash
python run_pipeline.py
```

### Individual Steps
```bash
python 01_generate_data.py      # ~2s — generates dataset
python 02_sql_analysis.py       # ~3s — runs SQL queries
python 03_ml_pipeline.py        # ~30s — full ML pipeline
python 04_generate_report.py    # ~5s — generates report
```

Then open `outputs/Telecom_Churn_Analysis_Report.html` in any browser.

---

## 🗃️ SQL Queries (10 Analytical Queries)

| # | Query | Business Question |
|---|-------|-------------------|
| 1 | Avg Call Duration by Churn | Are churners less engaged on calls? |
| 2 | Complaint Count by Churn | How do complaints drive attrition? |
| 3 | Recharge Frequency by Churn | Is recharge behavior a churn signal? |
| 4 | Contract Type Churn Rate | Which contract has highest churn? |
| 5 | Monthly Revenue Loss | What's the financial impact of churn? |
| 6 | Churn by Internet Service | Does fiber optic cause more churn? |
| 7 | Churn by Payment Method | Do electronic check users churn more? |
| 8 | Tenure Bucket Analysis | When in the lifecycle do customers leave? |
| 9 | Senior Citizen Analysis | Do seniors need specialized plans? |
| 10 | High-Value Customer Churn | What's the premium segment impact? |

---

## 🤖 ML Pipeline

### Models Trained
| Model | Purpose |
|-------|---------|
| **Logistic Regression** | Interpretable baseline — coefficients explain churn drivers |
| **Random Forest** | High-accuracy ensemble — best for production scoring |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, AUC-ROC
- Confusion Matrix (visual)
- ROC Curve comparison

### Class Imbalance
- **SMOTE** (Synthetic Minority Oversampling) applied to training data
- Recall (catching churners) prioritized over precision

---

## 🔍 Explainability

- **Feature Importance** — Random Forest built-in feature importances
- **SHAP** — SHapley Additive exPlanations for individual predictions
  - Summary plot (beeswarm)
  - Bar plot (mean absolute impact)

---

## 👥 Customer Segments

| Segment | Churn Prob | Strategy |
|---------|-----------|----------|
| 🔴 **At Risk** | > 50% | Immediate retention call + contract upgrade offer |
| 🟡 **Dormant** | 20-50% | Proactive engagement + auto-recharge incentives |
| 🟢 **Loyal** | < 20% | VIP rewards + upsell / referral program |

---

## 💡 Key Business Recommendations

1. **Contract Conversion** — Migrate Month-to-Month to annual plans (15-20% discount)
2. **24-hr Complaint SLA** — Fast-track resolution for 2+ complaint customers
3. **First-Year Program** — Welcome calls at Day 7, 30, 90 for new subscribers
4. **Auto-Recharge Discount** — ₹50/month off for auto-pay setup
5. **Value-Add Bundling** — Bundle OnlineSecurity + TechSupport at 30% discount
6. **ML Scoring in CRM** — Weekly batch scoring, flag >50% churn for outreach

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Python · pandas · numpy |
| SQL | SQLite (in-memory) · 10 analytical queries |
| ML | scikit-learn · imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer) |
| Visualization | matplotlib · seaborn |
| Report | Self-contained HTML/CSS |

---

## 📄 License

This project is part of the Data Analyst Internship at Elevate Labs.
