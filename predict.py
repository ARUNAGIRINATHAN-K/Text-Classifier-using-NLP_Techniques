import joblib

# Load the model
model = joblib.load('model/text_model.pkl')

def predict_category(text):
    return model.predict([text])[0]
