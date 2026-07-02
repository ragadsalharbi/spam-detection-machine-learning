import os
import re
import string
import joblib
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Download NLTK stopwords (first run only)
nltk.download("stopwords")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Clean and preprocess text."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


print("Loading dataset...")

# Make sure spam.csv is in the project root
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Clean missing values
df.dropna(inplace=True)

# Preprocess messages
df["message"] = df["message"].apply(clean_text)

# Convert labels to numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Dataset loaded successfully!")
