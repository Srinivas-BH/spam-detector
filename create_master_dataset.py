import pandas as pd
import numpy as np

print("="*60)
print("CREATING MASTER SPAM DETECTION DATASET")
print("="*60)

datasets = []

# 1. Load ham_spam_emails.csv (234K emails with processed text)
print("\n1. Loading ham_spam_emails.csv...")
try:
    df1 = pd.read_csv('ham_spam_emails.csv')
    # Rename columns and normalize
    if 'processed_message' in df1.columns:
        df1 = df1.rename(columns={'processed_message': 'Message'})
        df1['Message'] = df1['Message'].fillna('')
    else:
        print("   Warning: processed_message column not found")
        df1 = None
except Exception as e:
    print(f"   Error: {e}")
    df1 = None

if df1 is not None:
    # Normalize labels
    if 'label' in df1.columns:
        df1['Category'] = df1['label'].map({0: 'ham', 1: 'spam'})
    datasets.append(df1[['Category', 'Message']])
    print(f"   Loaded {len(df1)} messages")

# 2. Load new_spam_data.csv (83K emails)
print("\n2. Loading new_spam_data.csv...")
try:
    df2 = pd.read_csv('new_spam_data.csv')
    # Rename and normalize
    if 'text' in df2.columns:
        df2 = df2.rename(columns={'text': 'Message'})
        df2['Message'] = df2['Message'].fillna('')
    if 'label' in df2.columns:
        df2['Category'] = df2['label'].map({0: 'ham', 1: 'spam'})
    datasets.append(df2[['Category', 'Message']])
    print(f"   Loaded {len(df2)} messages")
except Exception as e:
    print(f"   Error: {e}")

# 3. Load spam_data.csv (83K emails - may be duplicate of new_spam_data)
print("\n3. Loading spam_data.csv...")
try:
    df3 = pd.read_csv('spam_data.csv')
    if 'Category' in df3.columns and 'Message' in df3.columns:
        df3['Message'] = df3['Message'].fillna('')
        datasets.append(df3[['Category', 'Message']])
        print(f"   Loaded {len(df3)} messages")
except Exception as e:
    print(f"   Error: {e}")

# Combine all datasets
print("\n4. Combining datasets...")
if datasets:
    df_combined = pd.concat(datasets, ignore_index=True)
    print(f"   Total before deduplication: {len(df_combined)} messages")
    
    # Remove duplicates based on message content
    df_combined = df_combined.drop_duplicates(subset=['Message'], keep='first')
    print(f"   Total after deduplication: {len(df_combined)} messages")
    
    # Remove empty messages
    df_combined = df_combined[df_combined['Message'].str.len() > 0]
    print(f"   Total after removing empty messages: {len(df_combined)} messages")
    
    # Show distribution
    print("\n5. Final Distribution:")
    print(df_combined['Category'].value_counts())
    
    # Shuffle the dataset
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the master dataset
    output_file = 'master_spam_dataset.csv'
    df_combined.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n[SUCCESS] Saved {len(df_combined)} messages to {output_file}")
    
    # Calculate statistics
    spam_pct = (df_combined['Category'] == 'spam').sum() / len(df_combined) * 100
    ham_pct = (df_combined['Category'] == 'ham').sum() / len(df_combined) * 100
    print(f"\nDataset Statistics:")
    print(f"  Spam: {spam_pct:.1f}%")
    print(f"  Ham:  {ham_pct:.1f}%")
    print(f"  Balanced: {'Yes' if 40 < spam_pct < 60 else 'No'}")
    
else:
    print("\n[ERROR] No datasets were loaded successfully")

print("\n" + "="*60)

