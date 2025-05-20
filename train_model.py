import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("dataset/dataset1.csv")  # Ensure this file is in the same directory

# Define features and labels
TEXT_COLUMN = "Text"
LABEL_COLUMN = "oh_label"

# Ensure correct data types
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df[TEXT_COLUMN], df[LABEL_COLUMN], test_size=0.2, random_state=42)

# Initialize and fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train LinearSVC model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Save vectorizer and model
with open("tfidfvectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

with open("LinearSVCTuned.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model training complete. Vectorizer and model saved!")
