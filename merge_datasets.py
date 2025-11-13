import pandas as pd
import sys

# This is the name of the new file you downloaded from Kaggle.
# Make sure your file name matches this EXACTLY.
NEW_FILENAME = 'new_spam_data.csv'

# Output files (we save both for compatibility)
OUTPUT_SPAM_DATA = 'spam_data.csv'
OUTPUT_MASTER = 'master_spam_dataset.csv'

print("Starting dataset merge...")
print("="*30)

# --- 1. Load your ORIGINAL dataset (More robustly) ---
try:
    df_old = pd.read_csv('spam_data.csv', encoding='latin-1')
    
    # Check for original SMS dataset columns (v1, v2)
    if 'v1' in df_old.columns and 'v2' in df_old.columns:
        print("-> Detected old 'spam_data.csv' (v1, v2 format). Renaming columns.")
        df_old = df_old.rename(columns={'v1': 'Category', 'v2': 'Message'})
        # Keep only the columns we need, ignore the extra blank ones
        df_old = df_old[['Category', 'Message']]
        
    # Check for already-processed format
    elif 'Category' in df_old.columns and 'Message' in df_old.columns:
        print("-> Detected old 'spam_data.csv' (Category, Message format).")
        df_old = df_old[['Category', 'Message']]
        
    else:
        print("Error: Could not find 'Category'/'Message' or 'v1'/'v2' columns in spam_data.csv.")
        sys.exit() # Exit the script
        
    print(f"-> Loaded {len(df_old)} messages from spam_data.csv")
    
except Exception as e:
    print(f"-> Error reading old spam_data.csv: {e}")
    print("-> Continuing without old data...")
    df_old = pd.DataFrame(columns=['Category', 'Message'])

# --- 2. Load the NEW dataset ---
print("\n" + "="*30)
try:
    print(f"-> Looking for new dataset: '{NEW_FILENAME}'...")
    df_new = pd.read_csv(NEW_FILENAME, encoding='latin-1')
    
    if 'text' in df_new.columns and 'label' in df_new.columns:
        # This is for the "Spam Email Classification Dataset"
        print("-> Found new file ('text'/'label' format). Processing...")
        df_new.rename(columns={'text': 'Message', 'label': 'Category'}, inplace=True)
        df_new['Category'] = df_new['Category'].map({0: 'ham', 1: 'spam'})
        
    elif 'v1' in df_new.columns and 'v2' in df_new.columns:
        # This is for the "SMS Spam Collection Dataset"
        print("-> Found new file ('v1'/'v2' format). Processing...")
        df_new.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)
        
    else:
        print(f"FATAL ERROR: The new CSV file '{NEW_FILENAME}' doesn't have the expected columns.")
        print("Please use one of the datasets I recommended in the previous step.")
        sys.exit() # Exit the script

    df_new = df_new[['Category', 'Message']]
    print(f"-> Loaded {len(df_new)} messages from {NEW_FILENAME}")

except FileNotFoundError:
    print("\n" + "!"*30)
    print(f"FATAL ERROR: Could not find the file '{NEW_FILENAME}'.")
    print(f"Please make sure you have downloaded the correct RAW TEXT dataset and")
    print(f"saved it in the same folder as this script with that exact name.")
    print("!"*30)
    sys.exit() # Exit the script
    
except Exception as e:
    print(f"Error reading {NEW_FILENAME}: {e}")
    sys.exit() # Exit the script

# --- 3. Combine and Clean ---
print("\n" + "="*30)
print("-> Combining datasets...")
# Combine the old and new dataframes
df_combined = pd.concat([df_old, df_new], ignore_index=True)

print(f"-> Total messages before cleaning: {len(df_combined)}")

# Remove any duplicate messages
df_combined.drop_duplicates(subset=['Message'], inplace=True)
print(f"-> Total messages after removing duplicates: {len(df_combined)}")

# Add your specific phishing example that failed
phishing_example = {
    'Category': 'spam',
    'Message': 'Your account has been temporarily locked due to unusual activity. Please verify your identity immediately to avoid permanent suspension: http://bit.ly/secure-auth-123'
}
df_combined = pd.concat([df_combined, pd.DataFrame([phishing_example])], ignore_index=True)
print("-> Added your custom phishing example.")

# --- 4. Save the FINAL master dataset ---
# Save in both formats: spam_data.csv (legacy) and master_spam_dataset.csv (used by training)
df_combined.to_csv(OUTPUT_SPAM_DATA, index=False, encoding='utf-8')
df_combined.to_csv(OUTPUT_MASTER, index=False, encoding='utf-8')

print("\n" + "="*30)
print("SUCCESS!")
print(f"Saved {len(df_combined)} total unique messages to {OUTPUT_SPAM_DATA} and {OUTPUT_MASTER}.")
print("You can now train the models with:\n  python train_ensemble_model.py")
print("Then run the app with:\n  python app.py")