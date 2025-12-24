import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Dataset Understanding
# ---------------------------------------------------------
print("--- 1. Dataset Understanding ---")
# Load the dataset
try:
    df = pd.read_csv('loan_approval_dataset.csv')
except FileNotFoundError:
    print("Error: loan_approval_dataset.csv not found.")
    exit()

# Handle common issue in this dataset: leading/trailing spaces in column names and values
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()

print(f"Dataset Shape: {df.shape}")
print(f"Column Names: {df.columns.tolist()}")
print("\nData Types:")
print(df.dtypes)

print("\nClass Balance of loan_status:")
print(df['loan_status'].value_counts(normalize=True))

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------
print("\n--- 2. Data Preprocessing ---")
# Drop loan_id from features
df_ml = df.drop('loan_id', axis=1)

# Handle missing values (though usually this dataset has none)
df_ml = df_ml.dropna()

# Encode categorical input columns
le_education = LabelEncoder()
df_ml['education'] = le_education.fit_transform(df_ml['education'])

le_self_employed = LabelEncoder()
df_ml['self_employed'] = le_self_employed.fit_transform(df_ml['self_employed'])

# Encode target variable separately
le_target = LabelEncoder()
df['loan_status'] = le_target.fit_transform(df['loan_status']) # For EDA plots
df_ml['loan_status'] = le_target.fit_transform(df_ml['loan_status'])

# ---------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------
print("\n--- 3. Feature Engineering ---")

# Total_Assets
df_ml['Total_Assets'] = (df_ml['residential_assets_value'] + 
                         df_ml['commercial_assets_value'] + 
                         df_ml['luxury_assets_value'] + 
                         df_ml['bank_asset_value'])

# Income_to_Loan_Ratio
# Handle division by zero just in case
df_ml['Income_to_Loan_Ratio'] = df_ml['income_annum'] / df_ml['loan_amount']

# Dependents_Risk: Grouped as 0-1 (Low), 2-3 (Medium), 4+ (High)
def get_dependents_risk(n):
    if n <= 1: return 'Low'
    elif n <= 3: return 'Medium'
    else: return 'High'

df_ml['Dependents_Risk'] = df_ml['no_of_dependents'].apply(get_dependents_risk)

# Credit_Risk_Category based on cibil_score
def get_credit_risk(score):
    if score >= 750: return 'Excellent'
    elif score >= 650: return 'Good'
    elif score >= 550: return 'Average'
    else: return 'Poor'

df_ml['Credit_Risk_Category'] = df_ml['cibil_score'].apply(get_credit_risk)

# Encode the newly created categories for modelling
le_dep = LabelEncoder()
df_ml['Dependents_Risk_Encoded'] = le_dep.fit_transform(df_ml['Dependents_Risk'])

le_cred = LabelEncoder()
df_ml['Credit_Risk_Category_Encoded'] = le_cred.fit_transform(df_ml['Credit_Risk_Category'])

print("Feature engineering complete. New columns added.")

# ---------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
print("\n--- 4. Exploratory Data Analysis ---")

# Setting the style
sns.set(style="whitegrid")

# Create figure for plots
plt.figure(figsize=(15, 12))

# Plot 1: CIBIL Score vs Loan Status
plt.subplot(2, 2, 1)
sns.boxplot(x='loan_status', y='cibil_score', data=df)
plt.title('CIBIL Score vs Loan Status')
plt.xticks([0, 1], ['Approved', 'Rejected'])

# Plot 2: Income vs Loan Status
plt.subplot(2, 2, 2)
sns.boxplot(x='loan_status', y='income_annum', data=df)
plt.title('Income vs Loan Status')
plt.xticks([0, 1], ['Approved', 'Rejected'])

# Plot 3: Loan Amount vs Loan Status
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='loan_amount', hue='loan_status', kde=True, element="step")
plt.title('Loan Amount Distribution by Status')

# Plot 4: Correlation Heatmap (numerical columns)
plt.subplot(2, 2, 4)
num_cols = df_ml.select_dtypes(include=[np.number]).columns
sns.heatmap(df_ml[num_cols].corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_plots.png')
print("EDA plots saved as 'eda_plots.png'.")

# ---------------------------------------------------------
# 5. Model Building
# ---------------------------------------------------------
print("\n--- 5. Model Building ---")

# Prepare features and target
# Drop the original categorical strings and use the encoded ones
X = df_ml.drop(['loan_status', 'Dependents_Risk', 'Credit_Risk_Category'], axis=1)
y = df_ml['loan_status']

# Split dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression (Scaling required)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)

# Random Forest (Unscaled data)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Models trained successfully.")

# ---------------------------------------------------------
# 6. Model Evaluation
# ---------------------------------------------------------
print("\n--- 6. Model Evaluation ---")

def evaluate_model(model, X_te, y_te, name, is_scaled=False):
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds)
    rec = recall_score(y_te, preds)
    cm = confusion_matrix(y_te, preds)
    
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return acc

log_acc = evaluate_model(log_model, X_test_scaled, y_test, "Logistic Regression")
rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# ---------------------------------------------------------
# 7. Model Output & Explanation
# ---------------------------------------------------------
print("\n--- 7. Model Output & Explanation ---")

# Best Model selection
best_model = rf_model if rf_acc > log_acc else log_model
best_name = "Random Forest" if rf_acc > log_acc else "Logistic Regression"
print(f"Best Model selected: {best_name}")

# Feature Importance (for Random Forest)
if best_name == "Random Forest":
    importances = rf_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=False)
    
    print("\nTop 5 Important Features for Decision:")
    print(feat_importances.head(5))

# Example Prediction
sample_idx = 0
sample_data = X_test.iloc[[sample_idx]]
# Approval probability score is generated using the predict_proba() method
sample_pred = rf_model.predict(sample_data)[0]
sample_prob = rf_model.predict_proba(sample_data)[0]

decision = "Approved" if sample_pred == 0 else "Rejected"
print(f"\nExample Prediction for applicant at index {X_test.index[sample_idx]}:")
print(f"Decision: {decision}")
print(f"Probability Score: {sample_prob[sample_pred]:.4f}")
print("Explanation: The decision is heavily influenced by the CIBIL score and assets value.")

# ---------------------------------------------------------
# 8. Saving Model and Encoders for Prediction
# ---------------------------------------------------------
print("\n--- 8. Saving Model and Encoders ---")

# Create a directory for models if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the model
joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.joblib'))

# Save the encoders
joblib.dump(le_education, os.path.join(models_dir, 'le_education.joblib'))
joblib.dump(le_self_employed, os.path.join(models_dir, 'le_self_employed.joblib'))
joblib.dump(le_dep, os.path.join(models_dir, 'le_dep.joblib'))
joblib.dump(le_cred, os.path.join(models_dir, 'le_cred.joblib'))
joblib.dump(le_target, os.path.join(models_dir, 'le_target.joblib'))

print(f"Model and encoders saved in '{models_dir}/' directory.")

