import re
import string
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Load saved model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def clean_text(text):
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


st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam Detection System")
st.write("Classify SMS messages as Spam or Ham using Machine Learning.")

message = st.text_area(
    "Enter your message:",
    placeholder="Congratulations! You have won a free prize..."
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(message)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham (Not Spam)")
