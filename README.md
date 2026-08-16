# Predictive Modeling and Risk Scoring for Bank Customer Churn

**Finance Analyst Internship Project | Unified Mentor Private Limited**
**Intern:** Samruddhi Raut

## 📌 Project Overview
Customer churn directly impacts a bank's revenue, customer lifetime value,
and long-term competitiveness. This project analyzes 10,000 European bank
customer records to identify the key drivers of customer churn and builds
a machine learning model that predicts churn risk before it happens. The
final model is deployed as an interactive Streamlit web application so
that relationship managers can score any customer's churn risk in real time.

## 📊 Key Findings (EDA)
- Overall churn rate: **~20.4%** of customers exited.
- **Germany** has a noticeably higher churn rate than France or Spain.
- **Older customers** and customers with **only 1 product** churn more.
- Customers who are **not active members** churn far more than active ones.
- Balance and credit score also show meaningful relationships with churn.

## 🤖 Model Performance
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.70 | 0.38 | 0.71 | 0.49 | 0.77 |
| **Random Forest (final model)** | **0.82** | **0.54** | **0.71** | **0.61** | **0.86** |

The Random Forest model was selected as the final model because it has the
highest ROC-AUC (ability to separate churners from non-churners) while
keeping recall reasonably high (catching most at-risk customers).

## 🚀 How to Run Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Re-train the model (optional — trained artifacts are already included)
python notebooks/train_model.py

# 3. Launch the app
python -m streamlit run app.py
```

## 🌐 Live Deployment
This app is deployed on **Streamlit Community Cloud**.
https://predictive-modeling-europian-bank-sr.streamlit.app/

## 🧠 Business Recommendations
The bank should prioritize retention efforts on customers who are:
- Located in **Germany**
- **Not active members**
- Holding **only one product**
- **Older (45+)**

Offering loyalty perks, proactive check-ins, or product bundling to this
segment is likely to reduce churn most cost-effectively.

## 🛠️ Tech Stack
- Python (pandas, NumPy)
- scikit-learn (Logistic Regression, Random Forest)
- Matplotlib / Seaborn (visualization)
- Streamlit (web app deployment)
- GitHub (version control)

---
**Author:** Saket Jha
**Role:** Data Analyst Intern
**Organization:** Unified Mentor Private Limited
