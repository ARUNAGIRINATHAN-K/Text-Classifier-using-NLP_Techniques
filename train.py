import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from utils.preprocessing import load_and_preprocess_data, vectorize_text

def train_model(data_path, model_path, vectorizer_path):
    """Train a text classification model and save it."""
    df = load_and_preprocess_data(data_path)
    
    X = df['cleaned_text']
    y = df['Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}, Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    train_model('data/sample_data.csv', 'model/text_model.pkl', 'model/vectorizer.pkl')