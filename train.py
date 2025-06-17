import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv("data/Facebook.csv")

# Rename columns for convenience (optional)
df = df.rename(columns={"content": "Text", "score": "Label"})

# Drop missing values
df.dropna(subset=["Text", "Label"], inplace=True)

# Features and target
X = df["Text"]
y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/text_model.pkl")
print("✅ Model trained and saved as model/text_model.pkl")
