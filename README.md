# Gym Membership Retention & Incentive Strategy Model

> Churn prediction and risk segmentation for 25,000 gym members with scenario analysis quantifying the ROI of targeted retention incentives.

---

## Project Overview

Member churn is one of the most expensive problems a gym can face, acquiring a new member costs 5 - 7x more than retaining one. This project builds an end-to-end churn modeling pipeline that moves beyond simple rule-based segmentation to ML- driven risk cohorts, enabling smarter, tier-specific retention campaigns.

**Why it's relevant beyond fitness:** The same methodology - behavioral feature engineering, risk segmentation, and intervention scenario modeling. This applies directly to any subscription or recurring revenue product (marketplaces, SaaS, support operations). The question "which users are about to disengage, and what's the cheapest intervention?" is universal.

---

## Dataset

- **Source:** [Kaggle — Gym Customers Features and Churn](https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn)
- **Scale:** Expanded to 25,000 members via synthetic augmentation to reflect a realistic mid-size gym chain population
- **Churn rate:** ~18.7%
- **Key fields:** Age, contract period, group visit participation, class attendance frequency (current + historical), additional charges, membership lifetime, app logins, promo source, partner membership, proximity to location

---

## 🔧 Feature Engineering

Raw features were augmented with four engineered signals designed to capture behavioral dynamics that static snapshots miss:

| Feature | Definition | Rationale |
|---|---|---|
| `attendance_trend` | Current freq − historical avg freq | Direction of engagement change matters more than level |
| `spend_per_month` | Additional charges / lifetime months | Normalises spend for tenure differences |
| `engagement_score` | Weighted composite of attendance, app logins, group visits | Single signal capturing multi-channel engagement |
| `contract_short` | Binary flag: 1-month contract | Strongest structural churn predictor |
| `high_engagement` | Binary: engagement_score > 3 | Non-linear threshold effect |
| `improving_attendance` | Binary: attendance_trend > 0 | Captures recovery signal |

**SQL equivalent provided in notebook** - all feature engineering logic is documented as CTE-based SQL queries alongside the pandas implementation, reflecting a production-style pipeline where features are built upstream before model training.

---

## Modeling

### Baseline — Logistic Regression
Trained on raw features only. Serves as the interpretable benchmark.

| Metric | Score |
|---|---|
| Accuracy | 0.836 |
| AUC-ROC | 0.820 |
| Precision | 0.635 |

### Improved — Random Forest with Feature Engineering
Trained on full engineered feature set. Random forests capture the non-linear interaction effects (e.g., short contract + declining attendance) that logistic regression cannot.

| Metric | Score |
|---|---|
| Accuracy | 0.830 |
| AUC-ROC | 0.801 |
| Precision | 0.604 |

### Top Churn Drivers (Feature Importance)
1. `avg_class_freq_current` — 15.6%
2. `lifetime_months` — 14.5%
3. `engagement_score` — 13.3%
4. `avg_additional_charges` — 11.5%
5. `spend_per_month` — 10.6%

---

## Risk Segmentation

Using RF churn probabilities, members are assigned to four actionable cohorts. Segmentation accuracy is measured against a naive baseline (attendance quartile splits) using within-segment churn variance - lower variance = purer, more actionable segments.

| Segment | Members | Churn Rate | Action |
|---|---|---|---|
| Low Risk | 16,375 | 1.4% | Reduce retention spend |
| Medium Risk | 5,219 | 29.5% | Monitor + light nudges |
| High Risk | 2,819 | 84.5% | Targeted discount offer |
| Critical | 533 | 97.2% | Immediate outreach |

**Segmentation accuracy improvement over naive baseline: 30.6%** (within-segment variance reduction)

---

## 💡 Scenario Analysis

### Scenario 1 — Discounted Renewal Offer (High Risk + Critical)
Concentrated discount offers on the 3,352 at-risk members project a **+7% lift in overall renewal rate** (81.3% → 87.0%), equivalent to ~230 additional retained members per campaign cycle.

### Scenario 2 — Reduced Promo Spend on Low Risk Members
Low risk members (16,375) are unlikely to churn without intervention. Reducing promotional spend by 10% for this cohort saves **~$4,094 per campaign cycle** while preserving retention outcomes.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn (LogisticRegression, RandomForestClassifier) |
| Visualization | matplotlib, seaborn |
| Feature pipeline | SQL (documented) + pandas |
| Environment | Google Colab / Jupyter |

---

## 📁 Repository Structure

```
gym-membership-retention/
│
├── gym_membership_retention.ipynb   # Full analysis notebook
├── README.md                        # This file
└── images/
    ├── eda_gym.png                  # 6-panel EDA visualization
    ├── model_comparison.png         # ROC curves + feature importance
    ├── segmentation.png             # Risk cohort analysis
    └── scenario_analysis.png        # Incentive strategy projections
```

---

## How to Run

1. Clone this repository
2. Download the base dataset from [Kaggle](https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn) (optional - notebook generates synthetic data by default)
3. Open `gym_membership_retention.ipynb` in Jupyter or Google Colab
4. Run all cells sequentially

---

## Author

**Likhita Nallapati**
[LinkedIn](https://linkedin.com/in/nlikhita3)
