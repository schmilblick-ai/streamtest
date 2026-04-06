import streamlit as st
from backend.loader import  load_embeddings #, load_w2v
from backend.analogy import compute_analogy
from backend.utils import load_css

load_css()
#model = load_w2v()
weights, idx_to_word, tokenizer = load_embeddings()

st.title("Analogies sémantiques")
st.caption("Logique : A - B + C = ?")

col1, col2, col3 = st.columns(3)
with col1:
    pos1 = st.text_input("A  (+)", value="great")
with col2:
    neg1 = st.text_input("B  (−)", value="story")
with col3:
    pos2 = st.text_input("C  (+)", value="acting")

topn = st.slider("Nombre de mots", 1, 10, 5)

if st.button("Calculer l'analogie ↗"):
    #WV results = compute_analogy(model, pos1, pos2, neg1)
    results = compute_analogy(pos1, pos2, neg1, tokenizer, weights, topn)
    if results:
        st.subheader(f"{pos1} − {neg1} + {pos2} ≈")
        for word, score in results:
            st.metric(label=word, value=f"{score:.3f}")
    else:
        st.error("Un des mots est absent du vocabulaire")