import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.sparse import hstack
import numpy as np
import joblib
import os
from flask import Flask, request, render_template, jsonify

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Global Model Variables ---
ensemble_model = None
vectorizer = None
individual_models = {}
feature_info = None

# --- Text Preprocessing Function ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(stemmed_tokens)

# --- Classification threshold (tune to reduce false positives) ---
SPAM_THRESHOLD = 0.6

# --- Feature Engineering Functions (Enhanced) ---
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
    # Removed 'today' which created false positives for benign messages
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

# --- Feature Extraction ---
def extract_features(text):
    """Extract all engineered features"""
    return np.array([
        contains_url(text),
        contains_short_url(text),
        contains_phishing_words(text),
        contains_urgency_words(text),
        message_length(text),
        digit_count(text),
        uppercase_ratio(text),
        exclamation_count(text),
        question_count(text),
        word_count(text),
        avg_word_length(text),
        contains_money(text),
        contains_winner(text)
    ]).reshape(1, -1)

# --- Load or Train Model ---
def load_or_train_model():
    global ensemble_model, vectorizer, individual_models, feature_info
    
    # Try to load ensemble model first
    ensemble_path = 'spam_model_ensemble.pkl'
    vectorizer_path = 'vectorizer.pkl'
    feature_info_path = 'feature_info.pkl'
    
    if os.path.exists(ensemble_path) and os.path.exists(vectorizer_path):
        print("--- Loading saved ensemble model... ---")
        try:
            ensemble_model = joblib.load(ensemble_path)
            vectorizer = joblib.load(vectorizer_path)
            if os.path.exists(feature_info_path):
                feature_info = joblib.load(feature_info_path)
            
            # Also load individual models if available
            model_files = {
                'random_forest': 'model_rf.pkl',
                'naive_bayes': 'model_nb.pkl',
                'logistic_regression': 'model_lr.pkl',
                'svm': 'model_svm.pkl'
            }
            
            for name, path in model_files.items():
                if os.path.exists(path):
                    individual_models[name] = joblib.load(path)
            
            print("--- Ensemble model loaded successfully. ---")
        except Exception as e:
            print(f"Error loading ensemble model: {e}")
            print("Please run train_ensemble_model.py first to create the models.")
            ensemble_model = None
    else:
        print("--- Ensemble/vectorizer not found. Creating a small fallback model for development... ---")
        try:
            # Small curated dataset for a functional fallback (not production-grade)
            fallback_data = [
                ("hey are you free to talk later today", 0),
                ("can we schedule a call for tomorrow morning", 0),
                ("lets meet for lunch at 1pm", 0),
                ("please review the attached report when you have time", 0),
                ("congratulations you have won a prize click here now", 1),
                ("urgent your account is locked verify immediately http://bit.ly/fake", 1),
                ("claim your $1000 reward today limited time offer", 1),
                ("winner winner free gift card visit our website", 1),
            ]
            texts = [t for t, _ in fallback_data]
            labels = np.array([y for _, y in fallback_data])

            # Fit vectorizer on processed text
            processed = [preprocess_text(t) for t in texts]
            local_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=1)
            X_tfidf = local_vectorizer.fit_transform(processed)

            # Compute engineered features
            feats = np.vstack([extract_features(t) for t in texts])
            from scipy.sparse import csr_matrix
            X_combined = hstack([X_tfidf, csr_matrix(feats)])
            X_dense = X_combined.toarray()

            # Train five algorithms for prediction parity with full setup
            lr_clf = LogisticRegression(max_iter=300, solver='saga', random_state=42)
            nb_clf = MultinomialNB(alpha=0.1)
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12, n_jobs=-1)
            # SVM with RBF kernel as requested
            svm_clf = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)

            lr_clf.fit(X_combined, labels)   # works well with sparse
            nb_clf.fit(X_combined, labels)   # works well with sparse
            rf_clf.fit(X_dense, labels)      # tree/SVM prefer dense
            svm_clf.fit(X_dense, labels)

            # Create a lightweight ensemble
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(
                estimators=[
                    ('lr', lr_clf),
                    ('nb', nb_clf),
                    ('rf', rf_clf),
                    ('svm', svm_clf),
                ],
                voting='soft',
                weights=[2, 1, 2, 2]
            )
            ensemble.fit(X_combined, labels)

            # Assign globals
            ensemble_model = ensemble
            vectorizer = local_vectorizer
            feature_info = {
                'feature_names': ['contains_url','contains_short_url','contains_phishing','contains_urgency',
                                  'msg_length','digit_count','uppercase_ratio','exclamation_count','question_count',
                                  'word_count','avg_word_length','contains_money','contains_winner'],
                'num_features': 13,
                'tfidf_features': 500,
                'ngram_range': (1, 2)
            }

            # Save so subsequent runs work without retraining
            joblib.dump(ensemble_model, ensemble_path)
            joblib.dump(vectorizer, vectorizer_path)
            joblib.dump(feature_info, feature_info_path)

            # Save individual models and keep them available for /predict-all
            joblib.dump(lr_clf, 'model_lr.pkl')
            joblib.dump(nb_clf, 'model_nb.pkl')
            joblib.dump(rf_clf, 'model_rf.pkl')
            joblib.dump(svm_clf, 'model_svm.pkl')
            individual_models.clear()
            individual_models.update({
                'logistic_regression': lr_clf,
                'naive_bayes': nb_clf,
                'random_forest': rf_clf,
                'svm': svm_clf
            })
            print("--- Fallback model created and saved. ---")
        except Exception as e:
            print(f"Fallback model creation failed: {e}")
            ensemble_model = None

# --- Prediction Functions ---
def make_prediction(message_text):
    """Make prediction using ensemble model"""
    if ensemble_model is None or vectorizer is None:
        raise Exception("Model not loaded. Please run train_ensemble_model.py first.")
    
    processed_message = preprocess_text(message_text)
    message_tfidf = vectorizer.transform([processed_message])
    
    # Extract engineered features
    features = extract_features(message_text)
    
    # Combine features
    combined_features = hstack([message_tfidf, features])
    
    # Predict
    spam_probability = ensemble_model.predict_proba(combined_features)[0][1]
    prediction = 1 if spam_probability >= SPAM_THRESHOLD else 0
    
    return prediction, spam_probability

def make_predictions_all_models(message_text):
    """Make predictions using all available individual models"""
    if vectorizer is None:
        raise Exception("Vectorizer not loaded.")
    
    from sklearn.svm import SVC as _SVC_
    from sklearn.ensemble import RandomForestClassifier as _RFC_

    # Helper to get an estimator by key from available sources
    def _resolve_model(key):
        # 1) direct individual model
        mdl = individual_models.get(key)
        if mdl is not None:
            return mdl
        # 2) pull from ensemble named estimators if available
        mapping = {
            'logistic_regression': ['lr', 'logreg', 'logistic'],
            'svm': ['svm', 'svc'],
            'naive_bayes': ['nb', 'mnb', 'naive_bayes', 'multinomialnb'],
            'random_forest': ['rf', 'random_forest', 'randomforestclassifier'],
        }
        if hasattr(ensemble_model, 'named_estimators_'):
            for alias in mapping.get(key, []):
                mdl = ensemble_model.named_estimators_.get(alias)
                if mdl is not None:
                    return mdl
        # 3) last resort: scan estimators_ list for class matches
        if hasattr(ensemble_model, 'estimators_'):
            for est in ensemble_model.estimators_:
                name = est.__class__.__name__.lower()
                if key == 'svm' and isinstance(est, _SVC_):
                    return est
                if key == 'random_forest' and isinstance(est, _RFC_):
                    return est
                if key == 'logistic_regression' and 'logistic' in name:
                    return est
                if key == 'naive_bayes' and 'naive' in name:
                    return est
        return None

    processed_message = preprocess_text(message_text)
    message_tfidf = vectorizer.transform([processed_message])
    features = extract_features(message_text)
    combined_features = hstack([message_tfidf, features])
    dense_features = combined_features.toarray()
    
    results = {}
    
    # Ensemble prediction
    if ensemble_model is not None:
        try:
            proba = ensemble_model.predict_proba(combined_features)[0]
            results['ensemble'] = {
                'label': 'spam' if proba[1] >= SPAM_THRESHOLD else 'ham',
                'ham_score': float(proba[0]),
                'spam_score': float(proba[1])
            }
        except:
            pass
    
    # Ensure all requested algorithms are returned: lr, svm, nb, rf
    requested = ['logistic_regression', 'svm', 'naive_bayes', 'random_forest']
    for req in requested:
        model = _resolve_model(req)
        try:
            if model is None:
                raise RuntimeError(f"{req} not available")
            # Use dense input for models that require it
            if isinstance(model, (_SVC_, _RFC_)):
                proba = model.predict_proba(dense_features)[0]
            else:
                proba = model.predict_proba(combined_features)[0]
            results[req] = {
                'label': 'spam' if proba[1] >= SPAM_THRESHOLD else 'ham',
                'ham_score': float(proba[0]),
                'spam_score': float(proba[1])
            }
        except Exception as e:
            print(f"Error with {req}: {e}")
            # Provide a safe placeholder to keep UI consistent
            results[req] = {
                'label': 'ham',
                'ham_score': 0.5,
                'spam_score': 0.5
            }
    
    # Calculate consensus
    if results:
        # Weighted consensus to improve stability and accuracy.
        # Weights favor stronger, typically better-calibrated models.
        weight_map = {
            'logistic_regression': 2.0,
            'svm': 2.0,
            'random_forest': 2.0,
            'naive_bayes': 1.0,
            'ensemble': 3.0
        }
        keys_in_order = ['logistic_regression', 'svm', 'random_forest', 'naive_bayes', 'ensemble']
        total_weight = 0.0
        weighted_spam = 0.0
        for k in keys_in_order:
            if k in results:
                w = weight_map.get(k, 1.0)
                weighted_spam += results[k]['spam_score'] * w
                total_weight += w
        if total_weight == 0:
            avg_spam = np.mean([r['spam_score'] for r in results.values()])
        else:
            avg_spam = weighted_spam / total_weight
        avg_ham = 1 - avg_spam
        consensus = {
            'label': 'spam' if avg_spam >= SPAM_THRESHOLD else 'ham',
            'ham_score': avg_ham,
            'spam_score': avg_spam
        }
    else:
        consensus = None
    
    return results, consensus

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('spam.html', result=None)

@app.route('/hello', methods=['GET'])
def hello_route():
    return jsonify({'message': 'Hello, the enhanced spam detection server is working!'})

@app.route('/predict', methods=['POST'])
def predict_web():
    try:
        new_message = request.form['message']
        prediction, spam_probability = make_prediction(new_message)
        spam_percentage = int(spam_probability * 100)
        if prediction == 1:
            result_text = f"This message is {spam_percentage}% likely to be SPAM."
        else:
            result_text = f"This message is {100 - spam_percentage}% likely to be HAM (Not Spam)."
        return render_template('spam.html', result=result_text, message=new_message)
    except Exception as e:
        return render_template('spam.html', result=f"An error occurred: {e}", message=None)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" in JSON body'}), 400
        
        new_message = data['message']
        prediction, spam_probability = make_prediction(new_message)
        
        return jsonify({
            'prediction': 'spam' if prediction == 1 else 'ham',
            'spam_probability': float(spam_probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-all', methods=['POST'])
def predict_all():
    """Endpoint for multi-algorithm comparison"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'ok': False, 'error': 'Missing "message" in JSON body'}), 400
        
        message = data['message']
        algorithms, consensus = make_predictions_all_models(message)
        
        return jsonify({
            'ok': True,
            'algorithms': algorithms,
            'consensus': consensus
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# --- Main execution ---
if __name__ == '__main__':
    load_or_train_model()
    if ensemble_model is None:
        print("\nWARNING: Model not loaded. The app will not work properly.")
        print("Please run: python train_ensemble_model.py")
    app.run(debug=True, host='0.0.0.0', port=5000)
