
import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import random

st.set_page_config(page_title="LLM Sentiment Analyzer", layout="wide")

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

llm = load_llm()

st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

# Dataset (either static or dynamic)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded Successfully!")
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

# Visualization
st.title("Amazon Review LLM Analyzer")
st.markdown("Get sentiment, emotion, and summary analysis using a local language model (FLAN-T5).")
st.subheader("📊 Dataset Overview")

col1, col2 = st.columns([2, 3])
with col1:
    st.write("### Ratings Distribution")
    if "rating" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='rating', data=df, palette="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No 'rating' column in the dataset to show distribution.")

with col2:
    st.write("### Dataset Summary")
    st.write(f"Total Reviews: {len(df)}")
    if "rating" in df.columns:
        st.write(df['rating'].describe())
    st.dataframe(df.head(5))

st.markdown("---")

st.subheader("Sample Reviews ")
sample_reviews = random.sample(df['text'].dropna().tolist(), min(5, len(df)))
for i, review in enumerate(sample_reviews):
    with st.expander(f"Review {i+1}"):
        st.code(review)

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

# Run LLM 
if st.button("Ask the bot!"):
    if review_input.strip() == "":
        st.warning("Please paste a review first.")
    else:
        with st.spinner("Thinking..."):
            prompt = make_prompt(task, review_input)
            result = llm(prompt)[0]['generated_text']
        st.success("Done!")
        st.markdown("**LLM Response:**")
        st.markdown(f"```text\n{result.strip()}\n```")

