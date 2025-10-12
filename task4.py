import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function (same as in training)
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Function to predict
def predict_message(message):
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return 'spam' if prediction == 1 else 'ham'

# Test with sample messages
if __name__ == "__main__":
    test_messages = [
        "Congratulations! You've won a free ticket to Bahamas. Call now!",
        "Hey, are we meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Verify now.",
        "Hi mom, how are you doing?"
    ]

    for msg in test_messages:
        result = predict_message(msg)
        print(f"Message: {msg}")
        print(f"Prediction: {result}")
        print("-" * 50)
