import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .result-container {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------

@st.cache_resource
def load_all_artifacts():
    """Load and cache model/encoders for efficiency."""
    models_dir = 'models'
    try:
        artifacts = {
            'model': joblib.load(os.path.join(models_dir, 'rf_model.joblib')),
            'le_edu': joblib.load(os.path.join(models_dir, 'le_education.joblib')),
            'le_self': joblib.load(os.path.join(models_dir, 'le_self_employed.joblib')),
            'le_dep': joblib.load(os.path.join(models_dir, 'le_dep.joblib')),
            'le_cred': joblib.load(os.path.join(models_dir, 'le_cred.joblib')),
            'le_target': joblib.load(os.path.join(models_dir, 'le_target.joblib'))
        }
        return artifacts
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None

def get_dependents_risk(n):
    if n <= 1: return 'Low'
    elif n <= 3: return 'Medium'
    else: return 'High'

def get_credit_risk(score):
    if score >= 750: return 'Excellent'
    elif score >= 650: return 'Good'
    elif score >= 550: return 'Average'
    else: return 'Poor'

# ---------------------------------------------------------
# 2. Sidebar & Title
# ---------------------------------------------------------

st.title("💰 Loan Approval Prediction System")
st.markdown("Enter applicant details below to predict the loan status with high accuracy.")

# Load artifacts
art = load_all_artifacts()

if art:
    # ---------------------------------------------------------
    # 3. User Input Layout
    # ---------------------------------------------------------
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Info")
            no_of_dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0)
            st.caption("_Estimates household financial burden based on family size._")
            
            education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
            st.caption("_Indicates academic background as a proxy for employment stability._")
            
            self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
            st.caption("_Assesses whether income is from a steady salary or business._")
            
        with col2:
            st.subheader("Financial Info")
            income_annum = st.number_input("Annual Income ($)", min_value=0, value=5000000, step=100000)
            st.caption("_Represents total yearly earnings available for debt repayment._")
            
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=1000000, step=100000)
            st.caption("_The requested sum, determining financial risk and repayment scale._")
            
            loan_term = st.number_input("Loan Term (Months)", min_value=1, value=36)
            st.caption("_Duration of the loan, used to calculate long-term commitement risk._")
            
            cibil_score = st.slider("CIBIL Score", 300, 900, 700)
            st.caption("_Numerical summary of credit history and past reliability._")
            
        with col3:
            st.subheader("Assets Value")
            res_assets = st.number_input("Residential Assets", min_value=0, value=2000000, step=100000)
            st.caption("_Value of the applicant's home, serving as primary collateral._")
            
            bank_assets = st.number_input("Bank Assets", min_value=0, value=300000, step=100000)
            st.caption("_Liquid savings available to support immediate debt servicing._")

            with st.expander("Optional Supporting Assets"):
                comm_assets = st.number_input("Commercial Assets", min_value=0, value=1000000, step=100000)
                st.caption("_Owned business properties indicating additional wealth._")
                
                lux_assets = st.number_input("Luxury Assets", min_value=0, value=500000, step=100000)
                st.caption("_High-value possessions reflecting spending power._")

        submit = st.form_submit_button("Predict Loan Status")

    if submit:
        # ---------------------------------------------------------
        # 4. Processing & Feature Engineering
        # ---------------------------------------------------------
        
        # Create input dict
        raw_data = {
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': res_assets,
            'commercial_assets_value': comm_assets,
            'luxury_assets_value': lux_assets,
            'bank_asset_value': bank_assets
        }
        
        df = pd.DataFrame([raw_data])
        
        # Label Encoding (Title Case already handled by selectbox matching)
        df['education'] = art['le_edu'].transform(df['education'])
        df['self_employed'] = art['le_self'].transform(df['self_employed'])
        
        # Feature Engineering
        df['Total_Assets'] = res_assets + comm_assets + lux_assets + bank_assets
        df['Income_to_Loan_Ratio'] = income_annum / loan_amount if loan_amount != 0 else 0
        
        dep_risk = get_dependents_risk(no_of_dependents)
        cred_risk = get_credit_risk(cibil_score)
        
        df['Dependents_Risk_Encoded'] = art['le_dep'].transform([dep_risk])[0]
        df['Credit_Risk_Category_Encoded'] = art['le_cred'].transform([cred_risk])[0]
        
        # Prepare features (Drop non-encoded categoricals as per training)
        X = df # In current training script, we kept numerical columns and added encoded ones.
        # Wait, need to check training columns order exactly.
        # Training dropped: ['loan_status', 'Dependents_Risk', 'Credit_Risk_Category']
        # Let's ensure order matches X.columns from loan_prediction.py:147
        
        # ---------------------------------------------------------
        # 5. Prediction
        # ---------------------------------------------------------
        
        probs = art['model'].predict_proba(X)[0]
        prediction = art['model'].predict(X)[0]
        decision = art['le_target'].inverse_transform([prediction])[0]
        
        # Display Results
        st.divider()
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if decision == 'Approved':
                st.success(f"### Result: {decision}")
                st.metric("Approval Probability", f"{probs[0]*100:.2f}%")
            else:
                st.error(f"### Result: {decision}")
                st.metric("Rejection Probability", f"{probs[1]*100:.2f}%")
        
        with res_col2:
            st.subheader("Applicant Profile Strength")
            
            # Dynamic Feature Strength Calculation
            # 1. CIBIL Strength (300-900)
            cibil_strength = (cibil_score - 300) / 600
            
            # 2. Income/Loan Strength (Ratio > 2 is very strong)
            ratio = income_annum / loan_amount if loan_amount > 0 else 5
            ratio_strength = min(ratio / 3, 1.0) 
            
            # 3. Asset Coverage Strength (Assets > 2x Loan is strong)
            assets_total = res_assets + comm_assets + lux_assets + bank_assets
            coverage = assets_total / loan_amount if loan_amount > 0 else 5
            coverage_strength = min(coverage / 2.5, 1.0)
            
            # 4. Loan Term Strength (Shorter is usually safer)
            term_strength = 1 - (min(loan_term, 120) / 120)
            
            strengths = [cibil_strength, ratio_strength, coverage_strength, term_strength]
            labels = ["CIBIL Score", "Income/Loan Ratio", "Asset Coverage", "Loan Term"]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            # Use color gradient based on strength
            colors = ['#28a745' if s > 0.7 else '#ffc107' if s > 0.4 else '#dc3545' for s in strengths]
            ax.barh(labels, strengths, color=colors)
            ax.set_xlim(0, 1)
            ax.set_title("How strong is your profile? (0.0 to 1.0)")
            for i, s in enumerate(strengths):
                ax.text(s + 0.02, i, f"{s:.2f}", va='center', fontweight='bold')
            st.pyplot(fig)
            st.caption("_These bars change dynamically as you adjust your inputs._")

        # ---------------------------------------------------------
        # 6. Explanations & Suggestions
        # ---------------------------------------------------------
        with st.expander("Analysis & Suggestions"):
            if decision == 'Rejected':
                st.warning("Your application was rejected based on the following reasons:")
                
                # Detailed Analysis Points
                analysis_points = []
                
                # 1. CIBIL Analysis
                if cibil_score < 550:
                    analysis_points.append("❌ **Poor Credit Score**: Your score is significantly below the threshold. Most lenders require at least 650 for consideration.")
                elif cibil_score < 750:
                    analysis_points.append("⚠️ **Average Credit Score**: While not in the 'Poor' category, increasing your score above 750 would make you a 'Preferred' applicant.")
                
                # 2. Financial Stability
                if income_annum < loan_amount:
                    analysis_points.append(f"❌ **High Debt-to-Income**: Your annual income (${income_annum:,}) is less than the loan amount (${loan_amount:,}). This is considered high risk.")
                elif income_annum < loan_amount * 2:
                    analysis_points.append("⚠️ **Tight Loan Margin**: Your income is sufficient but doesn't leave much room for financial emergencies.")

                # 3. Asset Security
                total_assets = res_assets + comm_assets + lux_assets + bank_assets
                if total_assets < loan_amount:
                    analysis_points.append("❌ **Insufficient Asset Backing**: Your total valued assets do not fully cover the loan amount, leading to higher collateral risk.")
                
                # 4. Term & Dependents
                if loan_term > 60 and cibil_score < 600:
                    analysis_points.append("⚠️ **High Duration with Low Score**: Long-term loans are harder to approve for applicants with average credit scores.")
                
                if no_of_dependents > 4:
                     analysis_points.append("⚠️ **High Family Burden**: A high number of dependents indicates higher mandatory expenses, reducing repayment capacity.")

                for point in analysis_points:
                    st.markdown(point)
                
                st.divider()
                st.subheader("💡 Steps to Approval")
                st.info("""
                1. **Improve CIBIL**: Ensure no defaults and lower your credit card utilization for 6 months.
                2. **Reduce Loan Amount**: Requesting 20-30% less might move your application to 'Approved' status.
                3. **Increase Assets**: Show higher bank savings or include a co-applicant with assets.
                4. **Longer History**: If self-employed, ensure you have at least 2 years of documented business income.
                """)
            else:
                st.success("### Why you were approved:")
                reasons = []
                if cibil_score >= 750: reasons.append("- Your **Excellent CIBIL Score** indicates high reliability.")
                if income_annum > loan_amount * 2: reasons.append("- Your **Strong Income Base** easily covers the loan amount.")
                if (res_assets + bank_assets) > loan_amount: reasons.append("- Your **Solid Asset Portfolio** provides excellent collateral.")
                
                for r in reasons:
                    st.markdown(r)
                
                st.balloons()
                st.info("Maintain this financial profile to ensure quick processing of your disbursement.")

else:
    st.warning("Please run the training script (`loan_prediction.py`) first to generate the required model and encoder files.")

# Footer
st.markdown("---")
st.caption("Developed for College Project Demonstration • powered by Python & Random Forest")
