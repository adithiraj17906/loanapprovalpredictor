# 🏦 Loan Approval Prediction System

A machine learning–based system that predicts whether a loan application should be **Approved** or **Rejected** using applicant financial and personal information.  
The model is trained on historical loan data and provides real-time predictions through a user-friendly GUI.

---

## 🚀 Live Demo

👉 **https://loanapprovepred.streamlit.app/**

---

## 🎯 Project Objective

To build an accurate and interpretable machine learning model that predicts loan approval status based on applicant income, credit score, loan details, and asset information.

---

## 📊 Dataset Overview

The model is trained using a structured CSV dataset containing the following features:

- Number of Dependents  
- Education  
- Self Employed  
- Annual Income  
- Loan Amount  
- Loan Term  
- CIBIL Score  
- Residential Assets Value  
- Commercial Assets Value  
- Luxury Assets Value  
- Bank Asset Value  
- Loan Status (Target Variable)

---

## 🧠 Feature Engineering

The following derived features are created to improve prediction accuracy:

- **Total Assets** – Sum of all asset values  
- **Income-to-Loan Ratio** – Measures repayment capacity  
- **Dependents Risk** – Categorizes financial burden  
- **Credit Risk Category** – Derived from CIBIL score  

---

## 🤖 Machine Learning Models

- **Logistic Regression** (Baseline Model)  
- **Random Forest Classifier** (Final Model)

The Random Forest model was selected based on superior accuracy, precision, and recall.

---

## 🖥️ Application Features

- Real-time loan approval prediction using user input  
- Approval probability score  
- Feature importance visualization  
- Actionable suggestions for rejected applications  
- Modern dark-themed Streamlit interface  

---

## ⚙️ Technology Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  
- Matplotlib  

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
