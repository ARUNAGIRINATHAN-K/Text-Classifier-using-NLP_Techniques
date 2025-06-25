import joblib
from utils.preprocessing import clean_text, preprocess_text

def predict_text(text, model_path='model/text_model.pkl', vectorizer_path='model/vectorizer.pkl'):
    """Predict the category of input text."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)
    
    X_vec = vectorizer.transform([preprocessed_text])
    prediction = model.predict(X_vec)
    
    return prediction[0]

if __name__ == "__main__":
    sample_text = input("Enter text to classify: ")
    result = predict_text(sample_text)
    print(f"Predicted category: {result}")