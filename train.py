import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv('data/Facebook.csv')

# Change this to your actual column names
X = df['Text']
y = df['Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for preprocessing + classification
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/text_model.pkl')
print("Model trained and saved.")
