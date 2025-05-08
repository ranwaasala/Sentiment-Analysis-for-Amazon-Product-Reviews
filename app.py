import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import joblib
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="LLM Sentiment Analyzer", layout="wide")

# Load LLM
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm = load_llm()

# Load ML sentiment model and tools
@st.cache_resource
def load_sentiment_model():
    model = joblib.load(r"C:\Users\Ranwah\grad_depi\logistic_regression_model.pkl")
    vectorizer = joblib.load(r"C:\Users\Ranwah\grad_depi\tfidf_vectorizer.pkl")
    label_encoder = joblib.load(r"C:\Users\Ranwah\grad_depi\label_encoder.pkl")
    return model, vectorizer, label_encoder

sentiment_model, vectorizer, label_encoder = load_sentiment_model()

def predict_sentiment(text):
    features = vectorizer.transform([text])
    label = sentiment_model.predict(features)[0]
    label_name = label_encoder.inverse_transform([label])[0]
    descriptions = {
        "Negative": "Negative — The review expresses dissatisfaction or criticism.",
        "Neutral": "Neutral — The review is balanced or mixed in tone.",
        "Positive": "Positive — The review expresses satisfaction or praise."
    }
    return label_name, descriptions.get(label_name, "")

# Upload Section
st.sidebar.header("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

# Load Data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("Uploaded file must contain a 'text' column.")
        st.stop()
    st.sidebar.success("File uploaded successfully!")
else:
    st.sidebar.info("Using sample reviews...")
    df = pd.DataFrame({
        "text": [
            "I love this product, it's amazing!",
            "Terrible service. It came broken.",
            "Not good, not bad. Just okay.",
            "Helpful support but poor quality.",
            "Fantastic item! Highly recommend."
        ],
        "rating": [5, 1, 3, 2, 5]
    })

# App Title and Description
st.title("Amazon Review LLM Analyzer")
st.markdown("Get sentiment, emotion, and summary analysis using a local language model (FLAN-T5).")

# Dataset Overview
st.subheader("📊 Dataset Overview")
col1, col2 = st.columns([2, 3])

with col1:
    st.write("### Ratings Distribution")
    if "rating" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='rating', data=df, palette="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No 'rating' column found in the dataset.")

with col2:
    st.write("### Dataset Summary")
    st.write(f"**Total Reviews:** {len(df)}")
    if "rating" in df.columns:
        st.write(df['rating'].describe())
    st.dataframe(df.head(5))

st.markdown("---")

# Sample Review Sentiment Buttons
st.subheader("Sample Reviews with ML Sentiment")
if "sample_reviews" not in st.session_state:
    st.session_state.sample_reviews = random.sample(df['text'].dropna().tolist(), min(5, len(df)))

sample_reviews = st.session_state.sample_reviews

for i, review in enumerate(sample_reviews):
    with st.expander(f"Review {i + 1}"):
        st.code(review, language="text")
        if st.button(f"Analyze Sentiment for Review {i + 1}", key=f"sentiment_{i}"):
            
            label, explanation = predict_sentiment(review)
            st.markdown(f"**Sentiment Prediction:** `{label}`")
            st.info(f"{explanation} This classification was generated using a machine learning model — specifically, logistic regression.")

            st.markdown("Why this prediction?")
            st.write(
                "The logistic regression model analyzed the review text using TF-IDF (Term Frequency-Inverse Document Frequency), "
                "which captures important words and their relative importance. Based on these patterns, the model predicted this review as "
                f"**{label.lower()}**. Logistic regression is a type of statistical model that finds patterns between words and sentiment labels."
            )

            st.caption("All sentiment results here are powered by logistic regression, trained on labeled review data.")


st.markdown("---")

# Interactive LLM Tool
st.subheader("Talk to the AI about a Review")
review_input = st.text_area("Paste or type a review here:", height=120)

task = st.selectbox("Choose what you want to do:", [
    "Classify Sentiment",
    "Summarize Review",
    "Detect Bias or Fake",
    "Tag Emotions"
])

def make_prompt(task, review):
    if task == "Classify Sentiment":
        return f"What is the overall sentiment of this review? Respond only with 'positive', 'neutral', or 'negative'.\n\nReview: {review}"
    elif task == "Summarize Review":
        return f"Summarize this customer review in one short sentence:\n\n{review}"
    elif task == "Detect Bias or Fake":
        return (
            "You are a professional NLP model. Your job is to detect whether a review sounds biased, fake, or emotionally exaggerated.\n\n"
            f"Review: {review}\n\n"
            "Respond with a clear answer starting with either:\n"
            "- Yes, this review is biased/fake.\n"
            "- No, this review seems genuine.\n"
            "Be specific and provide a reason."
        )
    elif task == "Tag Emotions":
        return f"List up to 3 emotions expressed in this review. Be specific (e.g., frustration, joy, anxiety):\n\n{review}"

if st.button("Ask the Bot"):
    if review_input.strip() == "":
        st.warning("Please paste a review first.")
    else:
        with st.spinner("Thinking..."):
            prompt = make_prompt(task, review_input)
            result = llm(prompt, max_length=256)[0]['generated_text']
        st.success("Done!")
        st.markdown("### LLM Response")
        st.code(result.strip(), language="text")
