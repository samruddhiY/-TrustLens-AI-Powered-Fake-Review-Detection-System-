import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# === Force Dark Theme ===
st.set_page_config(page_title="🕵️ Fake Review Detector", layout="centered")

# ✅ Inject CSS targeting .stApp container
st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #f0f0f0;
        }
        .stTextInput > div > div > input,
        .stTextArea textarea {
            background-color: #1c1c1c;
            color: #ffffff;
            border: 1px solid #444;
        }
        .stButton > button {
            background-color: #333333;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stButton > button:hover {
            background-color: #555555;
        }
        .result-box {
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.1rem;
            margin-top: 1rem;
        }
        .real {
            background-color: #0f5132;
            color: #d1e7dd;
        }
        .fake {
            background-color: #842029;
            color: #f8d7da;
        }
    </style>
""", unsafe_allow_html=True)


# === Load the model and vectorizer ===
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# === App UI ===
st.title("🕵️ Fake Review Detector")
st.subheader("Analyze a product review and detect if it's likely **real** or **fake**.")
review = st.text_area("📝 Paste or type your review below:", height=150)

# === Prediction ===
if st.button("🔍 Detect Review"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        # === Feature Engineering ===
        length = len(review)
        exclaim_count = review.count('!')
        words = review.split()
        upper_ratio = sum(1 for w in words if w.isupper()) / len(words) if words else 0

        custom_features = np.array([[length, exclaim_count, upper_ratio]])
        custom_df = pd.DataFrame(custom_features, columns=['length', 'exclaim_count', 'upper_case_ratio'])
        custom_sparse = csr_matrix(custom_df.values)

        # === Vectorize and Predict ===
        text_vector = vectorizer.transform([review])
        full_input = hstack([text_vector, custom_sparse])
        prediction = model.predict(full_input)[0]
        confidence = model.predict_proba(full_input)[0][prediction] * 100

        # === Show Result ===
        st.markdown("---")
        if prediction == 1:
            st.markdown(
                f"<div class='result-box real'>🟢 This review is <strong>likely genuine</strong>.<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='result-box fake'>🔴 This review is <strong>likely fake</strong>.<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Logistic Regression & TF-IDF - Dark Theme ✨")
