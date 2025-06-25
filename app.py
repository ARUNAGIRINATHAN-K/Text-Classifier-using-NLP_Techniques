from flask import Flask, request, render_template
import joblib
from utils.preprocessing import clean_text, preprocess_text

app = Flask(__name__)

model = joblib.load('model/text_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)
    X_vec = vectorizer.transform([preprocessed_text])
    prediction = model.predict(X_vec)[0]
    return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)