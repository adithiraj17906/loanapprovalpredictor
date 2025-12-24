import pandas as pd

df = pd.read_csv('loan_approval_dataset.csv')
print("Columns:", df.columns.tolist())
print("\nFirst row:\n", df.iloc[0])
