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
                'gradient_boosting': 'model_gb.pkl'
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
        print("--- Ensemble model not found. ---")
        print("Please run train_ensemble_model.py first to create the models.")
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
    prediction = ensemble_model.predict(combined_features)[0]
    spam_probability = ensemble_model.predict_proba(combined_features)[0][1]
    
    return prediction, spam_probability

def make_predictions_all_models(message_text):
    """Make predictions using all available individual models"""
    if vectorizer is None:
        raise Exception("Vectorizer not loaded.")
    
    processed_message = preprocess_text(message_text)
    message_tfidf = vectorizer.transform([processed_message])
    features = extract_features(message_text)
    combined_features = hstack([message_tfidf, features])
    
    results = {}
    
    # Ensemble prediction
    if ensemble_model is not None:
        try:
            proba = ensemble_model.predict_proba(combined_features)[0]
            results['ensemble'] = {
                'label': 'spam' if proba[1] > 0.5 else 'ham',
                'ham_score': float(proba[0]),
                'spam_score': float(proba[1])
            }
        except:
            pass
    
    # Individual model predictions
    for name, model in individual_models.items():
        try:
            proba = model.predict_proba(combined_features)[0]
            results[name] = {
                'label': 'spam' if proba[1] > 0.5 else 'ham',
                'ham_score': float(proba[0]),
                'spam_score': float(proba[1])
            }
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Calculate consensus
    if results:
        avg_spam = np.mean([r['spam_score'] for r in results.values()])
        avg_ham = 1 - avg_spam
        consensus = {
            'label': 'spam' if avg_spam > 0.5 else 'ham',
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
