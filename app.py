import os
import joblib
import pandas as pd
import streamlit as st

# ---- Paths ----
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "models")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLF_PATH = os.path.join(MODEL_DIR, "classifier.pkl")

# ---- Load model & vectorizer ----
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    return vectorizer, clf

vectorizer, clf = load_artifacts()

def predict_with_probs(text: str):
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_
    return classes, probs

# ---- UI ----
st.set_page_config(page_title="Emotion-Aware Chatbot", page_icon="🎭")
st.title("🎭 Emotion-Aware Chatbot")
st.write("Type a sentence and I’ll detect the emotion.")

user_text = st.text_area("Your message:", height=120, placeholder="I am excited for my interview today!")

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        classes, probs = predict_with_probs(user_text)
        top_idx = probs.argmax()
        st.success(f"Detected emotion: **{classes[top_idx]}**")

        # Show bar chart
        df = pd.DataFrame({"Emotion": classes, "Probability": probs})
        st.bar_chart(df.set_index("Emotion"))
