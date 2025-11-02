"""
Quick Start Script - Check if everything is ready
"""
import os
import sys

print("="*60)
print("SPAM DETECTION SYSTEM - QUICK START CHECK")
print("="*60)

# Check if model files exist
model_files = [
    'spam_model_ensemble.pkl',
    'vectorizer.pkl',
    'feature_info.pkl'
]

print("\n1. Checking model files...")
all_models_exist = True
for file in model_files:
    exists = os.path.exists(file)
    status = "[OK]" if exists else "[MISSING]"
    print(f"   {status} {file}")
    if not exists:
        all_models_exist = False

if not all_models_exist:
    print("\n[WARNING] Some model files are missing!")
    print("Please run: python train_ensemble_model.py")
    print("\nDo you want to train the model now? (This will take 5-10 minutes)")
    response = input("Enter 'yes' to train, or 'no' to continue anyway: ").strip().lower()
    if response == 'yes':
        print("\nTraining model...")
        os.system('python train_ensemble_model.py')
    else:
        print("\nContinuing without training...")
        print("Note: The Flask app may not work properly without trained models.")
else:
    print("\n[SUCCESS] All model files found!")

# Check if Flask app exists
print("\n2. Checking Flask app...")
if os.path.exists('app.py'):
    print("   [OK] app.py found")
else:
    print("   [MISSING] app.py not found!")
    sys.exit(1)

# Check dependencies
print("\n3. Checking dependencies...")
try:
    import flask
    import pandas
    import sklearn
    import nltk
    import joblib
    print("   [OK] All required packages installed")
except ImportError as e:
    print(f"   [MISSING] Missing package: {e}")
    print("   Please run: pip install -r requirements.txt")
    sys.exit(1)

print("\n" + "="*60)
print("SETUP CHECK COMPLETE!")
print("="*60)

if all_models_exist:
    print("\n[SUCCESS] Everything is ready!")
    print("\nTo start the Flask app, run:")
    print("   python app.py")
    print("\nThen open your browser to:")
    print("   http://localhost:5000")
    print("\nOr test the API:")
    print("   python test_api.py")
else:
    print("\n⚠️  Model files missing - please train the model first")

