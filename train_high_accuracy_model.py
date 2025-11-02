"""
High-Accuracy Spam Detection Model Training
Target: 99% accuracy using larger dataset and optimized hyperparameters
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import joblib
import os

print("="*60)
print("TRAINING HIGH-ACCURACY SPAM DETECTION MODEL (Target: 99%)")
print("="*60)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("\nDownloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(stemmed_tokens)

# Feature engineering
def contains_url(text):
    return 1 if re.search(r'http[s]?://|www\.', text, re.IGNORECASE) else 0

def contains_short_url(text):
    return 1 if re.search(r'bit\.ly|t\.co|tinyurl|goo\.gl', text, re.IGNORECASE) else 0

def contains_phishing_words(text):
    words = ['lock', 'verifi', 'suspend', 'unusu', 'activ', 'password', 'secur', 'ident', 
             'click', 'urgent', 'immedi', 'expir', 'warn', 'risk', 'compr']
    text_processed = preprocess_text(text)
    return 1 if any(word in text_processed for word in words) else 0

def contains_urgency_words(text):
    words = ['immedi', 'urgent', 'now', 'avoid', 'act', 'quick', 'asap', 'today']
    text_processed = preprocess_text(text)
    return 1 if any(word in text_processed for word in words) else 0

def message_length(text):
    return len(text)

def digit_count(text):
    return sum(c.isdigit() for c in text)

def uppercase_ratio(text):
    if len(text) == 0:
        return 0
    return sum(c.isupper() for c in text) / len(text)

def exclamation_count(text):
    return text.count('!')

def question_count(text):
    return text.count('?')

def word_count(text):
    return len(text.split())

def avg_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return np.mean([len(word) for word in words])

def contains_money(text):
    return 1 if re.search(r'\$|\d+ dollars|\d+ pounds|prize|million|thousand', text, re.IGNORECASE) else 0

def contains_winner(text):
    return 1 if re.search(r'winner|won|prize|congrats|congratulations', text, re.IGNORECASE) else 0

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('master_spam_dataset.csv', encoding='utf-8')
print(f"   Loaded {len(df)} messages")

# Use larger sample for better accuracy (80K samples)
print("   Sampling dataset (80K samples for higher accuracy)...")
df_spam = df[df['Category'] == 'spam'].sample(n=min(40000, len(df[df['Category'] == 'spam'])), random_state=42)
df_ham = df[df['Category'] == 'ham'].sample(n=min(40000, len(df[df['Category'] == 'ham'])), random_state=42)
df = pd.concat([df_spam, df_ham], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   Using {len(df)} messages for training")

df.dropna(subset=['Message', 'Category'], inplace=True)
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
print("   Preprocessing text...")
df['processed_message'] = df['Message'].apply(preprocess_text)

X_text = df['processed_message']
Y = df['Category']

print(f"   Distribution: {Y.value_counts().to_dict()}")

# Feature engineering
print("\n2. Engineering features...")
features_df = pd.DataFrame()
features_df['contains_url'] = df['Message'].apply(contains_url)
features_df['contains_short_url'] = df['Message'].apply(contains_short_url)
features_df['contains_phishing'] = df['Message'].apply(contains_phishing_words)
features_df['contains_urgency'] = df['Message'].apply(contains_urgency_words)
features_df['msg_length'] = df['Message'].apply(message_length)
features_df['digit_count'] = df['Message'].apply(digit_count)
features_df['uppercase_ratio'] = df['Message'].apply(uppercase_ratio)
features_df['exclamation_count'] = df['Message'].apply(exclamation_count)
features_df['question_count'] = df['Message'].apply(question_count)
features_df['word_count'] = df['Message'].apply(word_count)
features_df['avg_word_length'] = df['Message'].apply(avg_word_length)
features_df['contains_money'] = df['Message'].apply(contains_money)
features_df['contains_winner'] = df['Message'].apply(contains_winner)

print(f"   Created {len(features_df.columns)} engineered features")

# TF-IDF Vectorization with more features
print("\n3. Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2, max_df=0.95)
X_tfidf_train = vectorizer.fit_transform(X_text)
X_train_combined = hstack([X_tfidf_train, features_df.values])

print(f"   Combined feature shape: {X_train_combined.shape}")

# Train-Test split
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_train_combined, Y, test_size=0.2, random_state=42, stratify=Y
)
print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Train optimized models
print("\n5. Training optimized models...")

models = {
    'random_forest': RandomForestClassifier(
        n_estimators=300, 
        random_state=42, 
        n_jobs=-1, 
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced'
    ),
    'extra_trees': ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=25,
        class_weight='balanced'
    ),
    'naive_bayes': MultinomialNB(alpha=0.1),
    'logistic_regression': LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        C=2.0, 
        solver='saga',
        n_jobs=-1,
        class_weight='balanced'
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=200, 
        random_state=42, 
        max_depth=10,
        learning_rate=0.05
    ),
}

model_scores = {}

print("\n   Training progress:")
for name, model in models.items():
    print(f"   - {name}...", end=" ")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_scores[name] = accuracy
        print(f"Accuracy: {accuracy:.4f}")
        
        # Save individual model
        model_file = f'model_{name[0:2] if len(name.split("_")) > 1 else name}.pkl'
        joblib.dump(model, model_file)
        print(f"      Saved to {model_file}")
    except Exception as e:
        print(f"Error: {e}")
        model_scores[name] = 0.0

# Create weighted ensemble
print("\n6. Creating optimized ensemble...")
sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
top_models = [m for m, _ in sorted_models[:4]]

print(f"   Selected models: {top_models}")

# Create weighted ensemble based on individual accuracies
weights = [model_scores[name] for name in top_models]
ensemble_models = [(name, models[name]) for name in top_models]
ensemble = VotingClassifier(estimators=ensemble_models, voting='soft', weights=weights)

print("   Training ensemble...", end=" ")
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"Accuracy: {ensemble_accuracy:.4f}")

# Evaluation
print("\n7. Model Evaluation:")
print("="*60)

print("\nIndividual Model Accuracies:")
for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {score:.4f} ({score*100:.2f}%)")

print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['HAM', 'SPAM']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_ensemble)
print(cm)

# Save models
print("\n8. Saving models...")
joblib.dump(ensemble, 'spam_model_ensemble.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

feature_info = {
    'feature_names': features_df.columns.tolist(),
    'num_features': len(features_df.columns),
    'tfidf_features': 5000,
    'ngram_range': (1, 3)
}
joblib.dump(feature_info, 'feature_info.pkl')

print("   Models saved:")
print("   - spam_model_ensemble.pkl (ensemble model)")
print("   - vectorizer.pkl (TF-IDF vectorizer)")
print("   - feature_info.pkl (feature metadata)")

# Final statistics
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nFinal Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
print(f"Target: 99.00%")
print(f"Gap: {99.0 - ensemble_accuracy*100:.2f}%")
print(f"\nTotal training samples: {X_train.shape[0]}")
print(f"Total test samples: {X_test.shape[0]}")
print(f"\nModel is ready for deployment!")

if ensemble_accuracy >= 0.99:
    print("\n[SUCCESS] Target accuracy of 99% achieved!")
elif ensemble_accuracy >= 0.95:
    print("\n[GOOD] High accuracy achieved (>95%). Consider fine-tuning for 99%.")
else:
    print("\n[NOTE] Consider training on full dataset or tuning hyperparameters for higher accuracy.")

