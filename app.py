from flask import Flask, request, jsonify, render_template
import nltk
nltk.data.path.append("nltk_data")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import re

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Minimal stopwords list
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)          # remove <br/> tags
    text = text.lower()                              # lowercase
    text = re.sub(r'[^a-z\s]', '', text)            # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    if review == "":
        return jsonify({"error": "No review provided"}), 400

    clean_review = preprocess_text(review)
    review_vector = vectorizer.transform([clean_review])
    pred_class = model.predict(review_vector)[0]
    pred_proba = model.predict_proba(review_vector)[0].max()
    sentiment = "positive" if pred_class == 1 else "negative"

    return jsonify({
        "review": review,
        "predicted_sentiment": sentiment,
        "probability": round(float(pred_proba), 4)
    })

if __name__ == "__main__":
    app.run(debug=True)