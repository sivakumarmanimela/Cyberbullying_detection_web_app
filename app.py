from flask import Flask, render_template, request
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load Stopwords
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Load Vectorizer and Model
with open("tfidfvectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)  # Load the entire TF-IDF vectorizer
model = pickle.load(open("LinearSVCTuned.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form["user_input"]

        if user_input.strip():  # Check if input is not empty
            transformed_input = vectorizer.transform([user_input])
            prediction_result = model.predict(transformed_input)[0]

            # Convert numerical output to readable text
            prediction = "Cyberbullying" if prediction_result == 1 else "Non-Cyberbullying"

    return render_template("index.html", prediction=prediction, user_input=user_input)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)