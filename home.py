import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the trained model
# pipe_lr = joblib.load(open("/Users/ritesh/Desktop/EmoTrack/model/text_emotion.pkl", "rb"))

# glove model
# pipe_lr = joblib.load(open("/Users/ritesh/Desktop/EmoTrack/model/text_emotion_glove.pkl", "rb"))

#BERT
pipe_lr = joblib.load(open("/Users/ritesh/Desktop/EmoTrack/model/text_emotion_bert.pkl", "rb"))


# Emoji dictionary for emotions
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# # Function to predict emotions
# def predict_emotions(docx):
#     results = pipe_lr.predict([docx])
#     return results[0]

# # Function to get prediction probabilities
# def get_prediction_proba(docx):
#     results = pipe_lr.predict_proba([docx])
#     return results


# # Load GloVe embeddings
# def load_glove_embeddings(filepath):
#     embeddings = {}
#     with open(filepath, "r", encoding="utf8") as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             vector = np.asarray(values[1:], dtype='float32')
#             embeddings[word] = vector
#     return embeddings

# glove_file = "/Users/ritesh/Desktop/EmoTrack/static/glove.6B.100d.txt"  # Replace with your GloVe file path
# glove_embeddings = load_glove_embeddings(glove_file)

# # Document embedding function
# def document_embedding(text, embeddings, embedding_dim=100):
#     words = text.split()
#     valid_vectors = [embeddings[word] for word in words if word in embeddings]
#     if valid_vectors:
#         return np.mean(valid_vectors, axis=0)
#     else:
#         return np.zeros(embedding_dim)

# # Updated prediction functions
# def predict_emotions(docx):
#     doc_embedding = document_embedding(docx, glove_embeddings, embedding_dim=100).reshape(1, -1)
#     results = pipe_lr.predict(doc_embedding)
#     return results[0]

# def get_prediction_proba(docx):
#     doc_embedding = document_embedding(docx, glove_embeddings, embedding_dim=100).reshape(1, -1)
#     results = pipe_lr.predict_proba(doc_embedding)
#     return results


# from here bert started bert
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to extract BERT embeddings for a document
def bert_embedding(text, tokenizer, model, max_len=512):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    # Use the [CLS] token's representation for classification
    return output.last_hidden_state[:, 0, :].squeeze().numpy()

# Example: Get BERT embedding for a document
doc_embedding = bert_embedding("This is a test sentence.", tokenizer, bert_model)


# BERT embeddings
def predict_emotions(docx):
    doc_embedding = bert_embedding(docx, tokenizer, bert_model).reshape(1, -1)
    results = pipe_lr.predict(doc_embedding)
    return results[0]

def get_prediction_proba(docx):
    doc_embedding = bert_embedding(docx, tokenizer, bert_model).reshape(1, -1)
    results = pipe_lr.predict_proba(doc_embedding)
    return results


def app():
    st.title("Emotion Detection")
    st.subheader("Detect Emotions Based on Episodic Activity")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability)}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)