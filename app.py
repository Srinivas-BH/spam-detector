from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression #linear regression is a machine learning algorithm that is used to predict a continuous value.
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import os


def build_and_train_model() -> Pipeline:
    # Minimal sample dataset (you can replace with a larger real dataset)
    messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "URGENT! You have won a 1 week FREE membership",
        "Congratulations! You've been selected for a cash prize",
        "Claim your free ringtone now",
        "Dear user, exclusive offer just for you",
        "Win a brand new iPhone now",
        "Call this number now to receive your prize",
        "Lowest price meds, buy now",
        "Limited time deal, click the link",
        "You have been chosen to receive a reward",
        "Hey, are we still on for lunch today?",
        "I'll be there in 10 minutes",
        "Can you send me the report by EOD?",
        "Let's catch up this weekend",
        "Meeting moved to 3pm, see you then",
        "Don't forget to bring the documents",
        "Thanks for your help yesterday",
        "Happy birthday! Have a great day",
        "See you at the gym later",
        "What time works best for the call?",
    ]
    # 1 for spam, 0 for ham
    labels = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    # Build a simple pipeline: TF-IDF -> Linear Regression
    pipeline: Pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("linreg", LinearRegression()),
    ])

    # Fit on all data to keep it simple for demo purposes
    pipeline.fit(messages, np.array(labels, dtype=float))
    return pipeline


app = Flask(__name__, static_folder=None)
model: Pipeline = build_and_train_model()


@app.route("/")
def index():
    # Serve the existing spam.html
    directory = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(directory, "spam.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({
                "ok": False,
                "error": "Message is required",
            }), 400

        # Linear regression outputs a continuous score; threshold at 0.5
        score = float(model.predict([message])[0])
        # Clamp score to [0,1] to avoid out-of-range values
        score_clamped = max(0.0, min(1.0, score))
        label = "spam" if score_clamped >= 0.5 else "ham"

        return jsonify({
            "ok": True,
            "label": label,
            "score": score_clamped,
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc),
        }), 500


if __name__ == "__main__":
    # Run the development server
    app.run(host="127.0.0.1", port=5000, debug=True)


