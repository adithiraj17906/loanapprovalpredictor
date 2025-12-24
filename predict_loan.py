import pandas as pd
import numpy as np
import joblib
import os

def load_artifacts():
    """Load the trained model and encoders."""
    models_dir = 'models'
    try:
        rf_model = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
        le_education = joblib.load(os.path.join(models_dir, 'le_education.joblib'))
        le_self_employed = joblib.load(os.path.join(models_dir, 'le_self_employed.joblib'))
        le_dep = joblib.load(os.path.join(models_dir, 'le_dep.joblib'))
        le_cred = joblib.load(os.path.join(models_dir, 'le_cred.joblib'))
        le_target = joblib.load(os.path.join(models_dir, 'le_target.joblib'))
        return rf_model, le_education, le_self_employed, le_dep, le_cred, le_target
    except FileNotFoundError as e:
        print(f"Error: Required artifact not found. {e}")
        print("Please run 'loan_prediction.py' first to generate models and encoders.")
        exit()

def get_dependents_risk(n):
    """Categorize dependents risk as done in training."""
    if n <= 1: return 'Low'
    elif n <= 3: return 'Medium'
    else: return 'High'

def get_credit_risk(score):
    """Categorize credit risk as done in training."""
    if score >= 750: return 'Excellent'
    elif score >= 650: return 'Good'
    elif score >= 550: return 'Average'
    else: return 'Poor'

def get_user_input():
    """Accept and validate applicant details from console."""
    print("\n--- Enter Applicant Details ---")
    try:
        data = {}
        data['no_of_dependents'] = int(input("Number of Dependents: "))
        data['education'] = input("Education (Graduate / Not Graduate): ").strip()
        data['self_employed'] = input("Self Employed (Yes / No): ").strip()
        data['income_annum'] = float(input("Annual Income: "))
        data['loan_amount'] = float(input("Loan Amount: "))
        data['loan_term'] = int(input("Loan Term (in months): "))
        data['cibil_score'] = int(input("CIBIL Score: "))
        data['residential_assets_value'] = float(input("Residential Assets Value: "))
        data['commercial_assets_value'] = float(input("Commercial Assets Value: "))
        data['luxury_assets_value'] = float(input("Luxury Assets Value: "))
        data['bank_asset_value'] = float(input("Bank Asset Value: "))
        return data
    except ValueError:
        print("Invalid input. Please enter numeric values for numerical fields.")
        return None

def predict_loan():
    rf_model, le_education, le_self_employed, le_dep, le_cred, le_target = load_artifacts()
    
    applicant_data = get_user_input()
    if not applicant_data:
        return

    # Convert to DataFrame
    df = pd.DataFrame([applicant_data])

    # 1. Preprocessing (Label Encoding)
    try:
        # Normalize to Title Case to match LabelEncoder (e.g. 'graduate' -> 'Graduate')
        df['education'] = le_education.transform([df['education'][0].title()])[0]
        df['self_employed'] = le_self_employed.transform([df['self_employed'][0].title()])[0]
    except ValueError as e:
        print(f"Error in categorical input: {e}")
        print(f"Expected one of: {le_education.classes_} for Education")
        print(f"Expected one of: {le_self_employed.classes_} for Self Employed")
        return

    # 2. Feature Engineering
    df['Total_Assets'] = (df['residential_assets_value'] + 
                          df['commercial_assets_value'] + 
                          df['luxury_assets_value'] + 
                          df['bank_asset_value'])
    
    df['Income_to_Loan_Ratio'] = df['income_annum'] / df['loan_amount']
    
    df['Dependents_Risk'] = df['no_of_dependents'].apply(get_dependents_risk)
    df['Credit_Risk_Category'] = df['cibil_score'].apply(get_credit_risk)
    
    # 3. Encoding engineered features
    df['Dependents_Risk_Encoded'] = le_dep.transform(df['Dependents_Risk'])
    df['Credit_Risk_Category_Encoded'] = le_cred.transform(df['Credit_Risk_Category'])

    # 4. Prepare for prediction (Match training feature order)
    # Training session dropped: ['loan_status', 'Dependents_Risk', 'Credit_Risk_Category']
    X = df.drop(['Dependents_Risk', 'Credit_Risk_Category'], axis=1)

    # Prediction
    prediction = rf_model.predict(X)[0]
    probabilities = rf_model.predict_proba(X)[0]
    
    # Reverse transform target
    decision = le_target.inverse_transform([prediction])[0]
    
    print("\n" + "="*40)
    print(f"--- LOAN PREDICTION RESULT ---")
    print(f"Decision: {decision}")
    print(f"Approval Probability: {probabilities[0]:.4f}") # 0 usually maps to Approved in this dataset
    print("="*40)

    # 5. Explanation and Suggestions
    show_explanation_and_suggestions(applicant_data, X, rf_model, decision)

def show_explanation_and_suggestions(raw_data, feature_df, model, decision):
    # Feature Importance based explanation
    importances = model.feature_importances_
    feature_names = feature_df.columns
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\nTop Factors Influencing this Decision:")
    for i in range(3):
        idx = sorted_idx[i]
        print(f"- {feature_names[idx]} (Importance: {importances[idx]:.4f})")

    if decision == 'Rejected' or decision == 1: # Adjusting for encoded value vs string
        print("\n--- Suggestions for Improvement ---")
        suggestions = []
        
        if raw_data['cibil_score'] < 700:
            suggestions.append("1. Improve your CIBIL score. A score above 750 significantly increases approval chances.")
        
        if raw_data['income_annum'] < raw_data['loan_amount'] / 2:
            suggestions.append("2. Consider reducing the loan amount or increasing your documented annual income.")
            
        assets_total = (raw_data['residential_assets_value'] + raw_data['commercial_assets_value'] + 
                        raw_data['luxury_assets_value'] + raw_data['bank_asset_value'])
        if assets_total < raw_data['loan_amount']:
             suggestions.append("3. Increasing your asset base (e.g. Bank Savings) can help provide better collateral security.")
        
        if not suggestions:
            suggestions.append("Your application is close to approval. Try reducing the loan amount or increasing the term.")
            
        for s in suggestions:
            print(s)
    else:
        print("\nYour application looks strong. Maintain your current credit profile for smooth processing.")

if __name__ == "__main__":
    predict_loan()
