import streamlit as st
import pickle
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. NLTK Data Path Setup (Crucial for Streamlit Cloud Deployment) ---

# Get the directory where the app.py script is located
script_dir = os.path.dirname(__file__)

# Define the relative path to the nltk_data folder (This folder MUST be in your repo)
nltk_data_path = os.path.join(script_dir, "nltk_data")

# Add the local path to NLTK's data search path
# This prevents the LookupError and DownloadError on Streamlit Cloud
nltk.data.path.append(nltk_data_path)

# Initialize Lemmatizer globally before model loading
lemmatizer = WordNetLemmatizer()


# --- 2. Load Models (Cached for Performance) ---

@st.cache_resource
def load_models():
    """Loads the pre-trained TF-IDF vectorizer and the Logistic Regression model."""
    try:
        # --- PATHS RESTORED TO ORIGINAL NAMES ---
        vectorizer_filepath = os.path.join(script_dir, 'tfidf_vectorizer.pkl')
        model_filepath = os.path.join(script_dir, 'logistic_regression_model.pkl')
        # --- END UPDATE ---

        # Load Vectorizer
        with open(vectorizer_filepath, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Load Model
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
            
        return tfidf_vectorizer, model
    except FileNotFoundError as e:
        # Update the error message to reflect the original filenames for debugging
        st.error(f"Error: Model file not found. Please ensure 'tfidf_vectorizer.pkl' and 'logistic_regression_model.pkl' are in the same directory as app.py. Details: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading. Details: {e}")
        return None, None

# Load the models and stopwords globally after NLTK path is set
tfidf_vectorizer, model = load_models()
stop_words = set(stopwords.words('english')) # This will now work as the NLTK path is set


# --- 3. Text Preprocessing Function ---

def preprocess_text(text):
    """
    Cleans and processes the input text for prediction.
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URL links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize (split into words)
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Rejoin words into a single string
    return " ".join(words)


# --- 4. Prediction Function ---

def predict_fake_news(text, vectorizer, model):
    """
    Vectorizes text and returns the model prediction and confidence.
    """
    if not vectorizer or not model:
        # Handle cases where model loading failed
        return "N/A", 0.0, "Model/Vectorizer not loaded due to an earlier error."

    # Preprocess the text
    processed_text = preprocess_text(text)
    
    if not processed_text:
        return "N/A", 0.0, "Input text is too short or empty after processing."

    # Vectorize the preprocessed text
    # The vectorizer expects an iterable (like a list) of strings
    vectorized_text = vectorizer.transform([processed_text])

    # Get the prediction (0 or 1)
    prediction = model.predict(vectorized_text)[0]
    
    # Get the confidence score (probability for the predicted class)
    confidence = model.predict_proba(vectorized_text)[0]
    
    # Map the numerical prediction to labels
    label = "REAL" if prediction == 0 else "FAKE"
    
    # Get the confidence percentage for the predicted label
    confidence_score = confidence[prediction] * 100

    return label, confidence_score, None


# --- 5. Streamlit App Layout ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Fake News Detector", layout="centered")

    # ... (Styling code omitted for brevity but is in the file)
    st.markdown(
        """
        <style>
        .title {
            font-size: 32px;
            font-weight: 700;
            color: #333333;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 18px;
            color: #666666;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .real-box {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #c3e6cb;
            font-size: 1.1em;
            font-weight: bold;
        }
        .fake-box {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #f5c6cb;
            font-size: 1.1em;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="title">📰 Fake News Detector (Text Analytics Project)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter a news article or headline to classify it as REAL or FAKE.</div>', unsafe_allow_html=True)

    # Input Text Area
    news_text = st.text_area(
        "Enter News Article Text:", 
        placeholder="Example: 'Aliens land in Mumbai and cure all diseases. Is this true?'",
        height=200,
        key="news_input"
    )

    # Analyze Button
    if st.button("Analyze News"):
        if not news_text:
            st.warning("Please enter some text to analyze.")
        else:
            # Display spinner while processing
            with st.spinner('Analyzing the news text...'):
                label, confidence_score, error_message = predict_fake_news(news_text, tfidf_vectorizer, model)

                if error_message:
                    st.error(f"Prediction failed: {error_message}")
                else:
                    confidence_percent = f"{confidence_score:.2f}%"
                    
                    if label == "REAL":
                        box_class = "real-box"
                        emoji = "✅"
                    else:
                        box_class = "fake-box"
                        emoji = "❌"
                        
                    st.markdown("---")
                    st.subheader("Model Prediction:")
                    st.markdown(
                        f"""
                        <div class="{box_class}">
                            {emoji} Model Prediction: <span style="font-size: 1.2em;">{label}</span>
                            <br>
                            🔬 Confidence: {confidence_percent}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.info("Note: This is an automated prediction based on a trained model and should not be used as the sole source for verifying information.")

if __name__ == "__main__":
    main()