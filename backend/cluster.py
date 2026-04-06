import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

def compute_tsne_WV(model, n_words=300):
    words = list(model.wv.index_to_key[:n_words])
    vectors = np.array([model.wv[w] for w in words])

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(vectors)

    return pd.DataFrame({
        'word': words,
        'x': coords[:, 0],
        'y': coords[:, 1]
    })

#directement sur la matrice de poid -> change l'appel par rapport à VW
def compute_tsne(tokenizer, weights, n_words=300):
    words = list(tokenizer.word_index.keys())[:n_words]
    indices = [tokenizer.word_index[w] for w in words]
    vecs = weights[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(vecs)

    return pd.DataFrame({
        'word': words,
        'x': coords[:, 0],
        'y': coords[:, 1]
    })