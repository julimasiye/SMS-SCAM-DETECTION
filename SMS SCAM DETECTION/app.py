import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "SMSSpamCollection"
MODEL_PATH = BASE_DIR / "sms_spam_model.pkl"

app = Flask(__name__)


def train_model():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "SMSSpamCollection not found. Put the dataset in the same folder as this Python file."
        )

    data = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"] != ""]
    data = data.drop_duplicates(subset=["label", "message"])

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model trained successfully")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate (ham)", "Spam"],
        zero_division=0,
    ))
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    joblib.dump(model, MODEL_PATH)
    return model


def load_or_train_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return train_model()
    return train_model()


MODEL = load_or_train_model()


def predict_sms(message):
    text = (message or "").strip()
    if not text:
        raise ValueError("Please enter an SMS message before checking.")

    prediction = MODEL.predict([text])[0]
    probabilities = MODEL.predict_proba([text])[0]
    confidence = max(probabilities)

    label = "Possible scam / spam" if prediction == "spam" else "Legitimate message"
    return label, float(confidence)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check", methods=["POST"])
def check_sms():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter an SMS message before checking."}), 400

    try:
        result, confidence = predict_sms(message)
        return jsonify({
            "result": result,
            "confidence": round(confidence * 100, 1),
            "message": message,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
