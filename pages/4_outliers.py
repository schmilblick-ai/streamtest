import streamlit as st
import plotly.graph_objects as go
from backend.loader import load_embeddings#, load_w2v
from backend.outlier import find_outlier, outlier_scores #, find_outlier_WV, outlier_scores_WV
from backend.utils import load_css

load_css()
#model = load_w2v()
weights, idx_to_word, tokenizer = load_embeddings()

st.title("Détection d'outliers")
st.caption("Quel mot n'appartient pas au groupe ?")

raw = st.text_input(
    "Liste de mots séparés par des virgules",
    value="good, excellent, brilliant, terrible"
)

words = [w.strip() for w in raw.split(',') if w.strip()]

if len(words) >= 3 and st.button("Analyser ↗"):
    #outlier = find_outlier_WV(model, words)
    #scores = outlier_scores_WV(model, words)
    outlier = find_outlier(words, tokenizer, weights)
    scores = outlier_scores(words, tokenizer, weights)

    col1, col2 = st.columns(2)
    with col1:
        st.error(f"Outlier détecté : **{outlier}**")

    with col2:
        fig = go.Figure(go.Bar(
            x=[s for _,s in scores],
            y=[w for w,_ in scores],
            orientation='h',
            marker_color=['#e74c3c' if w == outlier
                          else '#3498db' for w,_ in scores]
        ))
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)