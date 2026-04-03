import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Text Generator By Janani Patil",
    page_icon="@@",
    layout="centered"
)

# -------------------------------
# Load Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_all():
    model = load_model("TextGenerationModel1.keras")
    tokenizer = joblib.load("tokenizer.joblib")
    return model, tokenizer

model, tokenizer = load_all()

max_sequence_len = 20  

# -------------------------------
# Sampling Function
# -------------------------------
def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # safety
    if np.any(np.isnan(preds)):
        preds = np.ones_like(preds) / len(preds)

    return np.random.choice(len(preds), p=preds)

# -------------------------------
# Text Generation
# -------------------------------
def generate_text(seed_text, next_words=20, temperature=0.7):
    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        preds = model.predict(token_list, verbose=0)[0]

        predicted_index = sample_with_temperature(preds, temperature)

        output_word = tokenizer.index_word.get(predicted_index, "")

        seed_text += " " + output_word

    return seed_text


# -------------------------------
# UI Design
st.title(" AI Shakespeare Text Generator")
st.markdown("By Janani Patil")


# Input
seed = st.text_input("✍️ Enter starting text:", "To be or not to be")

# Controls
col1, col2 = st.columns(2)

with col1:
    num_words = st.slider("📏 Number of Words", 10, 100, 20)

with col2:
    temperature = st.slider("🔥 Creativity", 0.2, 1.5, 0.7)

# Example Button
if st.button("🎲 Try Example"):
    seed = "The king said"

# Generate Button
if st.button(" Generate Text"):
    with st.spinner("Generating..."):
        result = generate_text(seed, num_words, temperature)

    st.success("✅ Generated Text:")
    st.write(result)

# -------------------------------
# Sidebar (Impressive)
# -------------------------------
st.sidebar.title(" Model Info")
st.sidebar.write("Model: LSTM")
st.sidebar.write("Framework: TensorFlow / Keras")
st.sidebar.write("Dataset: Shakespeare")
st.sidebar.write("Vocabulary: 8000 words")

# -------------------------------
# How it Works
# -------------------------------
with st.expander("ℹ️ How it works"):
    st.write("""
    1. Input text is converted into sequences using tokenizer  
    2. LSTM model predicts next word  
    3. Temperature controls randomness  
    4. Words are generated step-by-step  
    """)

# Footer
st.write("---")
st.caption("Built with  using Streamlit")

