from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
import numpy as np
import re
import string

def advanced_text_preprocessing(text):
    """
    Advanced text preprocessing for better spam detection accuracy.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_prepare_data():
    """
    Loads the full Kaggle SMS dataset, cleans it, and returns the processed data.
    """
    try:
        # Step 1: Load the standard Kaggle dataset ('spam_data.csv')
        # We use latin-1 encoding and specify to only use the first two columns to ensure clean parsing.
        df = pd.read_csv('spam_data.csv', encoding='latin-1', usecols=[0, 1])
        df.columns = ['label', 'message']
        
        # Step 2: Clean the data thoroughly
        # Drop any rows with missing values (NaNs).
        df.dropna(inplace=True)
        # Filter the dataset to only include rows with 'spam' or 'ham' labels. This is a critical
        # step to prevent errors and ensure the model only learns from valid data.
        df = df[df['label'].isin(['spam', 'ham'])]
        
        # Prepare the data for the model with advanced preprocessing
        messages = [advanced_text_preprocessing(str(msg)) for msg in df['message']]
        labels = df['label'].map({'spam': 1, 'ham': 0}).tolist()
        print(f"Successfully loaded and processed {len(messages)} messages from the Kaggle dataset.")
        return messages, labels

    except FileNotFoundError:
        print("FATAL: Dataset file 'spam_data.csv' not found. Please ensure it is in the correct directory.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading or processing the data: {e}")
        exit()

def build_and_train_models():
    """
    Builds and trains multiple ML algorithms with enhanced features for maximum accuracy.
    Returns a dictionary of trained models.
    """
    messages, labels = load_and_prepare_data()
    
    # Enhanced feature extraction with multiple vectorizers
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True, 
        stop_words="english", 
        max_features=10000, 
        ngram_range=(1,3),
        min_df=1,
        max_df=0.95
    )
    
    count_vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1,2),
        min_df=1,
        max_df=0.95
    )
    
    # Define enhanced algorithms with better parameters
    algorithms = {
        'logistic_regression': Pipeline([
            ("tfidf", tfidf_vectorizer),
            ("classifier", LogisticRegression(random_state=42, solver='liblinear', C=1.0, max_iter=1000))
        ]),
        'naive_bayes': Pipeline([
            ("tfidf", tfidf_vectorizer),
            ("classifier", MultinomialNB(alpha=0.1))
        ]),
        'random_forest': Pipeline([
            ("tfidf", tfidf_vectorizer),
            ("classifier", RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, min_samples_split=2))
        ]),
        'svm': Pipeline([
            ("tfidf", tfidf_vectorizer),
            ("classifier", SVC(random_state=42, probability=True, C=1.0, kernel='rbf'))
        ]),
        'ensemble': Pipeline([
            ("tfidf", tfidf_vectorizer),
            ("classifier", VotingClassifier([
                ('lr', LogisticRegression(random_state=42, solver='liblinear')),
                ('nb', MultinomialNB(alpha=0.1)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
                ('svm', SVC(random_state=42, probability=True, C=1.0))
            ], voting='soft'))
        ])
    }
    
    # Train all models and calculate accuracy
    trained_models = {}
    for name, pipeline in algorithms.items():
        print(f"Training {name.replace('_', ' ').title()}...")
        pipeline.fit(messages, labels)
        
        # Calculate cross-validation score for accuracy assessment
        try:
            cv_scores = cross_val_score(pipeline, messages, labels, cv=5, scoring='accuracy')
            accuracy = cv_scores.mean()
            print(f"{name.replace('_', ' ').title()} training complete. CV Accuracy: {accuracy:.3f}")
        except:
            print(f"{name.replace('_', ' ').title()} training complete.")
        
        trained_models[name] = pipeline
    
    return trained_models

# Initialize the Flask web application and train all models on startup
app = Flask(__name__, static_folder=None)
models = build_and_train_models()

@app.route("/")
def index():
    """Serves the main HTML user interface."""
    directory = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(directory, "spam.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Receives a user's message and returns the exact prediction percentages from logistic regression."""
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Message is required"}), 400

        # Use the logistic regression model to predict probabilities for [ham, spam]
        probabilities = models['logistic_regression'].predict_proba([message])[0]
        ham_probability = float(probabilities[0])
        spam_probability = float(probabilities[1])

        # Determine the final label based on the higher probability
        label = "spam" if spam_probability > ham_probability else "ham"

        # Return a JSON response with the exact percentages
        return jsonify({
            "ok": True,
            "label": label,
            "spam_score": spam_probability,
            "ham_score": ham_probability,
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": f"An unexpected error occurred: {str(exc)}",
        }), 500

@app.route("/predict-all", methods=["POST"])
def predict_all():
    """Receives a user's message and returns predictions from all algorithms with percentages."""
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Message is required"}), 400

        # Preprocess the input message
        processed_message = advanced_text_preprocessing(message)
        
        # Get predictions from all models
        results = {}
        for name, model in models.items():
            probabilities = model.predict_proba([processed_message])[0]
            ham_probability = float(probabilities[0])
            spam_probability = float(probabilities[1])
            label = "spam" if spam_probability > ham_probability else "ham"
            
            results[name] = {
                "label": label,
                "spam_score": spam_probability,
                "ham_score": ham_probability
            }

        # Calculate average scores across all models
        avg_spam_score = np.mean([result["spam_score"] for result in results.values()])
        avg_ham_score = np.mean([result["ham_score"] for result in results.values()])
        consensus_label = "spam" if avg_spam_score > avg_ham_score else "ham"

        return jsonify({
            "ok": True,
            "consensus": {
                "label": consensus_label,
                "spam_score": float(avg_spam_score),
                "ham_score": float(avg_ham_score)
            },
            "algorithms": results
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": f"An unexpected error occurred: {str(exc)}",
        }), 500

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)

