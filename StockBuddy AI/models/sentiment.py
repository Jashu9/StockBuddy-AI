from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function: Tokenize, remove stopwords, and lemmatize
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
    return " ".join(filtered_tokens)


# Apply preprocessing to the 'title' column
news_df['processed_title'] = news_df['title'].apply(preprocess_text)

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Apply sentiment analysis on processed text
news_df["sentiment"] = news_df["processed_title"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
