import streamlit as st
import numpy as np
import joblib
import nltk

model = None
tokenizer = None
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(
    page_title="AI Text Generator By Janani Patil",
    page_icon="✨",
    layout="centered"
)


def get_model():
    global model, tokenizer
    if model is None:
        try:
            model = load_model("TextGenerationModel1.keras")
            tokenizer = joblib.load("tokenizer.joblib")
        except Exception as e:
            st.error("Model load nahi ho raha. Check file path / size.")
            st.write(e)
            st.stop()
    return model, tokenizer

max_sequence_len = 20  

def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype("float64")

    if np.sum(preds) == 0:
        preds = np.ones_like(preds) / len(preds)

    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    if np.any(np.isnan(preds)):
        preds = np.ones_like(preds) / len(preds)

    return np.random.choice(len(preds), p=preds)

def generate_text(seed_text, next_words=20, temperature=0.7):
    model, tokenizer = get_model()

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        preds = model.predict(token_list, verbose=0)[0]

        if np.sum(preds) == 0:
            continue

        predicted_index = sample_with_temperature(preds, temperature)

        output_word = tokenizer.index_word.get(predicted_index, "")

        if output_word == "":
            continue

        seed_text += " " + output_word

    return seed_text

st.title("AI Shakespeare Text Generator")
st.markdown("**By Janani Patil**")

seed = st.text_input("Enter starting text:", "To be or not to be")

col1, col2 = st.columns(2)

with col1:
    num_words = st.slider("Number of Words", 10, 100, 20)

with col2:
    temperature = st.slider("Creativity", 0.2, 1.5, 0.7)

if st.button("Generate Text"):

    if seed.strip() == "":
        st.warning("Please enter some text")
        st.stop()

    with st.spinner("Loading model..."):
        model, tokenizer = get_model()

    with st.spinner("Generating text..."):
        result = generate_text(seed, num_words, temperature)

    st.success("Generated Text:")
    st.write(result)

st.sidebar.title("Model Info")
st.sidebar.write("Model: LSTM")
st.sidebar.write("Framework: TensorFlow / Keras")
st.sidebar.write("Dataset: Shakespeare")
st.sidebar.write("Vocabulary: 8000 words")

with st.expander(" How it works"):
    st.write("""
    1. Input text is converted into sequences using tokenizer  
    2. LSTM model predicts next word  
    3. Temperature controls randomness  
    4. Words are generated step-by-step  
    """)

st.write("---")
st.caption("Built with using Streamlit")

