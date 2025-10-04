import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download NLTK resources if not already present
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# --- 1. Load the Model and Vectorizer ---
try:
    with open('models/tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    with open('models/logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'models/tfidf_vectorizer.pkl' and 'models/logistic_regression_model.pkl' exist.")
    st.stop()


# --- 2. Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans, tokenizes, removes stopwords, and lemmatizes the input text."""
    # Convert to lowercase and remove punctuation/special characters
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# --- 3. Prediction Function ---
def predict_news(news_text):
    """Preprocesses the text and uses the model to predict the class and confidence."""

    # Preprocess the input text
    cleaned_text = preprocess_text(news_text)

    # Vectorize the cleaned text (using the fitted TF-IDF vectorizer)
    text_vector = tfidf_vectorizer.transform([cleaned_text])

    # Predict the class (0 for Real, 1 for Fake)
    prediction = model.predict(text_vector)[0]

    # Get the confidence score (probability of the predicted class)
    # The output of predict_proba is [P(class 0), P(class 1)]
    probabilities = model.predict_proba(text_vector)[0]
    confidence = probabilities[prediction] * 100

    # Map the prediction to a human-readable label
    label = "FAKE" if prediction == 1 else "REAL"

    return label, confidence


# --- 4. Streamlit App Interface ---

st.title("📰 Fake News Detector (Text Analytics Project)")
st.markdown("Enter a news article or headline to classify it as **REAL** or **FAKE**.")

# Text area for user input
user_input = st.text_area("Enter News Article Text:", height=200, 
                          placeholder="Example: 'Aliens land in Mumbai and cure all diseases. Is this true?'")

# Prediction button
if st.button("Analyze News"):
    if user_input:
        # Get the prediction
        label, confidence = predict_news(user_input)

        st.markdown("---")
        st.subheader("Analysis Result")

        # Display the result with appropriate color/icon
        if label == "REAL":
            st.success(f"**Model Prediction:** {label} News")
        else:
            st.error(f"**Model Prediction:** {label} News")

        # Display the confidence score
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.markdown("---")
        st.markdown("*Note: This model is a demonstration and should not be used for critical fact-checking.*")

    else:
        st.warning("Please enter some news text to analyze.")