import streamlit as st
import plotly.express as px
from backend.loader import load_embeddings #,load_w2v
from backend.cluster import compute_tsne
from backend.utils import load_css
from SL_app01 import cinoch_header

cinoch_header()
load_css()
#model = load_w2v()
weights, idx_to_word, tokenizer = load_embeddings()

st.title("Visualisation TSNE des embeddings")
st.text("TSNE is t-distributed Stochastic Neighbor Embedding")
n_words = st.slider("Nombre de mots à visualiser", 50, 500, 200)

with st.spinner("Calcul TSNE en cours..."):
    #dftsne = compute_tsne_Wv(model, n_words)
    dftsne = compute_tsne(tokenizer, weights, n_words)

fig = px.scatter(
    dftsne, x='x', y='y', text='word',
    hover_name='word',
    width=900, height=600
)
fig.update_traces(textposition='top center', textfont_size=9)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)