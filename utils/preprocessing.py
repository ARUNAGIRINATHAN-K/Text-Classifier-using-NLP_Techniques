import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Clean text by removing punctuation, converting to lowercase, and removing special characters."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

def preprocess_text(text):
    """Preprocess text: tokenize, remove stopwords, and stem."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data(file_path):
    """Load dataset and preprocess text data."""
    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['Text'].apply(clean_text).apply(preprocess_text)
    return df

def vectorize_text(train_texts, test_texts=None):
    """Vectorize text using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)
        return X_train, X_test, vectorizer
    return X_train, vectorizer