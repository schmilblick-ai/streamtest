import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec

@st.cache_resource
def load_w2v():
    return Word2Vec.load('model.w2v')

@st.cache_resource
def load_model(mdlName="word2vec.keras"):
    mdlPath=Path(f"data/{mdlName}")
    return tf.keras.models.load_model(mdlPath)

@st.cache_resource
def load_df(dfName="MovieReview_gld.csv"):
    return pd.read_csv(Path(f"data/{dfName}"))

@st.cache_resource
def load_tokenizer(tknizer='tokenizer.pkl', df=False):
    #Ne pas faire fit_on_texts, riski, préfer sauver le tokenizer pendat la phase de model build
    if df:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(df.review)
    if tknizer:
        #préférer le pickel.load pour récupérer le tokenizer buildé pendant l'inférence
        with open(Path(f'data/{tknizer}'), 'rb') as f:
            return pickle.load(f)

@st.cache_resource
def load_embeddings():
    model = load_model()
    weights = model.get_layer('embedding').get_weights()[0]
    tokenizer = load_tokenizer()
    idx_to_word = {v: k for k, v in tokenizer.word_index.items()}
    return weights, idx_to_word, tokenizer    