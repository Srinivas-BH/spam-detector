from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# train_test_split is no longer needed as we train on the full dataset for production
import pandas as pd
import os

def build_and_train_model() -> Pipeline:
    """
    Loads the full Kaggle SMS dataset, cleans it, and trains the final model 
    on all available data for maximum accuracy.
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
        
        # Prepare the data for the model
        messages = df['message'].astype(str).tolist()
        labels = df['label'].map({'spam': 1, 'ham': 0}).tolist()
        print(f"Successfully loaded and processed {len(messages)} messages from the Kaggle dataset.")

    except FileNotFoundError:
        print("FATAL: Dataset file 'spam_data.csv' not found. Please ensure it is in the correct directory.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading or processing the data: {e}")
        exit()

    # Step 3: Build the machine learning pipeline.
    # This pipeline defines the steps to process the text and make a prediction.
    pipeline: Pipeline = Pipeline([
        # TfidfVectorizer: Converts text messages into meaningful numerical vectors.
        # We also tell it to ignore common English stop words.
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000, ngram_range=(1,2))),
        
        # LogisticRegression: A powerful and reliable classification algorithm for this task.
        ("classifier", LogisticRegression(random_state=42, solver='liblinear')),
    ])

    # Step 4: Train the final model on the ENTIRE dataset.
    # By using all 5,500+ messages for training, we ensure the model has learned
    # from every possible example, maximizing its predictive accuracy.
    print(f"Training the final model on all {len(messages)} messages for maximum accuracy...")
    pipeline.fit(messages, labels)
    print("Model training complete.")
    
    return pipeline

# Initialize the Flask web application and train the model on startup
app = Flask(__name__, static_folder=None)
model: Pipeline = build_and_train_model()

@app.route("/")
def index():
    """Serves the main HTML user interface."""
    directory = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(directory, "spam.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Receives a user's message and returns the exact prediction percentages."""
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Message is required"}), 400

        # Use the fully trained model to predict probabilities for [ham, spam]
        probabilities = model.predict_proba([message])[0]
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

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)

