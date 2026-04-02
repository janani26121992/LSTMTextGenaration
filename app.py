import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model (your file name)
model = load_model("TextGenerationModel1.keras")

# Load tokenizer (joblib)
tokenizer = joblib.load("tokenizer.joblib")

max_sequence_len = 20   # ⚠️ MUST match training


# 🔥 Temperature sampling
def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


# 🧠 Generate text
def generate_text(seed_text, next_words=20):
    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        preds = model.predict(token_list, verbose=0)[0]

        predicted_index = sample_with_temperature(preds, 0.7)

        output_word = tokenizer.index_word.get(predicted_index, "")

        seed_text += " " + output_word

    return seed_text


# 🎭 UI
st.title("Text Genaration \n By Janani Patil.")

seed = st.text_input("Enter starting text:", "To be or not to be")

num_words = st.slider("Words to generate", 10, 100, 20)

if st.button("Generate"):
    result = generate_text(seed, num_words)
    st.write(result)