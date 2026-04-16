import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from backend.loader import load_embeddings #, load_w2v
from backend.similar import get_similar_words, build_freq_dict
from backend.utils import load_css
from SL_app01 import cinoch_header

cinoch_header()

load_css()
#model = load_w2v()
weights, idx_to_word, tokenizer = load_embeddings()

title_placeholder = st.empty()

col1, col2 = st.columns([2, 1])
with col1:
    word = st.text_input("Entrez un mot", value="")
with col2:
    topn = st.slider("Nombre de mots", 10, 100, 30)

if word:
    title_placeholder.title(f"Mots conceptuellement proches de **{word}**")
    #similar = get_similar_words_WV(model, word, topn)
    similar = get_similar_words(word,tokenizer,weights,topn)
    if not similar:
        st.warning(f"'{word}' absent du vocabulaire")
    else:
        freq = build_freq_dict(similar)

        # Wordcloud
        wc = WordCloud(width=800, height=400,
                       background_color=None, mode="RGBA",
                       colormap='plasma').generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Table détaillée
        with st.expander("Voir les scores de similarité"):
            st.dataframe(
                {"Mot": [w for w,_ in similar],
                 "Similarité": [round(s, 4) for _,s in similar]},
                use_container_width=True
            )
else:
    title_placeholder.title(f"Mots conceptuellement proches - entrez un mot")