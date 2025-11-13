import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import joblib
import os

print("="*60)
print("TRAINING ENSEMBLE SPAM DETECTION MODEL")
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

# ============= TEXT PREPROCESSING =============
def preprocess_text(text):
    """Enhanced text preprocessing"""
    text = str(text).lower()
    # Remove special characters but keep URLs and punctuation for feature extraction
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(stemmed_tokens)

# ============= ADVANCED FEATURE ENGINEERING =============
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
    # Removed 'today' to avoid false positives on benign coordination messages
    words = ['immedi', 'urgent', 'now', 'avoid', 'act', 'quick', 'asap']
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

# ============= LOAD DATA =============
print("\n1. Loading dataset...")
df = pd.read_csv('master_spam_dataset.csv', encoding='utf-8')
print(f"   Loaded {len(df)} messages")

# Sample dataset for faster training (30K total for speed)
print("   Sampling dataset for faster training (30K samples)...")
df_spam = df[df['Category'] == 'spam'].sample(n=min(15000, len(df[df['Category'] == 'spam'])), random_state=42)
df_ham = df[df['Category'] == 'ham'].sample(n=min(15000, len(df[df['Category'] == 'ham'])), random_state=42)
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

# ============= FEATURE ENGINEERING =============
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

# ============= TF-IDF VECTORIZATION =============
print("\n3. Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)  # Reduced for speed
X_tfidf_train = vectorizer.fit_transform(X_text)
X_train_combined = hstack([X_tfidf_train, features_df.values])

print(f"   Combined feature shape: {X_train_combined.shape}")

# ============= TRAIN-TEST SPLIT =============
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_train_combined, Y, test_size=0.2, random_state=42, stratify=Y
)
print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# ============= TRAIN INDIVIDUAL MODELS =============
print("\n5. Training individual models...")

models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, verbose=0),
    'naive_bayes': MultinomialNB(alpha=0.1),
    'logistic_regression': LogisticRegression(random_state=42, max_iter=300, C=1.0, solver='saga', n_jobs=-1),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5, verbose=0),
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
        
        # Save individual model with proper naming
        name_mapping = {
            'random_forest': 'model_rf.pkl',
            'naive_bayes': 'model_nb.pkl',
            'logistic_regression': 'model_lr.pkl',
            'gradient_boosting': 'model_gb.pkl'
        }
        model_file = name_mapping.get(name, f'model_{name}.pkl')
        joblib.dump(model, model_file)
    except Exception as e:
        print(f"Error: {e}")
        model_scores[name] = 0.0

# ============= CREATE ENSEMBLE MODEL =============
print("\n6. Creating ensemble model...")

# Get top performing models for ensemble
sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
top_models = [m for m, _ in sorted_models[:3]]  # Top 3 models for faster ensemble

print(f"   Selected models for ensemble: {top_models}")

# Create ensemble with top models
ensemble_models = [(name, models[name]) for name in top_models]
ensemble = VotingClassifier(estimators=ensemble_models, voting='soft', weights=[2, 2, 1])  # Weight better models more

print("   Training ensemble...", end=" ")
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"Accuracy: {ensemble_accuracy:.4f}")

# Use best single model if it's better than ensemble
best_single_model_name = max(model_scores.items(), key=lambda x: x[1])[0]
best_single_accuracy = model_scores[best_single_model_name]

if best_single_accuracy > ensemble_accuracy:
    print(f"   Using best single model ({best_single_model_name}) instead of ensemble")
    ensemble = models[best_single_model_name]
    ensemble_accuracy = best_single_accuracy
    y_pred_ensemble = models[best_single_model_name].predict(X_test)

# ============= EVALUATION =============
print("\n7. Model Evaluation:")
print("="*60)

print("\nIndividual Model Accuracies:")
for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {score:.4f} ({score*100:.2f}%)")

print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['HAM', 'SPAM']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))

# ============= SAVE MODELS =============
print("\n8. Saving models...")

# Save ensemble as main model
joblib.dump(ensemble, 'spam_model_ensemble.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save feature engineering info
feature_info = {
    'feature_names': features_df.columns.tolist(),
    'num_features': len(features_df.columns),
    'tfidf_features': 2000,
    'ngram_range': (1, 2)
}
joblib.dump(feature_info, 'feature_info.pkl')

print("   Models saved:")
print("   - spam_model_ensemble.pkl (ensemble model)")
print("   - vectorizer.pkl (TF-IDF vectorizer)")
print("   - feature_info.pkl (feature metadata)")

# ============= FINAL STATISTICS =============
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nFinal Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
print(f"Total training samples: {X_train.shape[0]}")
print(f"Total test samples: {X_test.shape[0]}")
print(f"\nModel is ready for deployment!")

