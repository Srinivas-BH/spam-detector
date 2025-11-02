import pandas as pd
import numpy as np

print("="*60)
print("ANALYZING DATASETS")
print("="*60)

# 1. Analyze ham_spam_emails.csv
print("\n1. ham_spam_emails.csv")
try:
    df1 = pd.read_csv('ham_spam_emails.csv')
    print(f"   Total rows: {len(df1)}")
    print(f"   Columns: {df1.columns.tolist()}")
    if 'label' in df1.columns:
        print(f"   Label distribution: {df1['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"   Error: {e}")

# 2. Analyze new_spam_data.csv  
print("\n2. new_spam_data.csv")
try:
    df2 = pd.read_csv('new_spam_data.csv')
    print(f"   Total rows: {len(df2)}")
    print(f"   Columns: {df2.columns.tolist()}")
    if 'label' in df2.columns:
        print(f"   Label distribution: {df2['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"   Error: {e}")

# 3. Analyze spam_data.csv
print("\n3. spam_data.csv (current main dataset)")
try:
    df3 = pd.read_csv('spam_data.csv', nrows=100000)  # Sample
    print(f"   Total rows (sampled): {len(df3)}")
    print(f"   Columns: {df3.columns.tolist()}")
    if 'Category' in df3.columns:
        print(f"   Category distribution: {df3['Category'].value_counts().to_dict()}")
except Exception as e:
    print(f"   Error: {e}")

# 4. Analyze email_phishing_data.csv
print("\n4. email_phishing_data.csv")
try:
    df4 = pd.read_csv('email_phishing_data.csv', nrows=100)
    print(f"   Total rows: {len(df4)}")
    print(f"   Columns: {df4.columns.tolist()}")
    print(f"   First row sample: {df4.iloc[0].to_dict()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)


