from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#** Only valid with a gensim word2vec kind of model
def get_similar_words_WV(model, word, topn=30):
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return []

#with a tensorflow kind of model - thin
def get_similar_words(word, tokenizer, weights, topn=30):
    idx = tokenizer.word_index.get(word)
    if idx is None:
        return []
    vec = weights[idx]
    # Cosine similarity entre ce vecteur et tous les autres
    sims = cosine_similarity([vec], weights)[0]
    # Trier par similarité décroissante
    top_idx = np.argsort(sims)[::-1][1:topn+1]
    # Reconstruire mot depuis index
    idx_to_word = {v: k for k, v in tokenizer.word_index.items()}
    #renvoi les top tuples mots, et similarity
    return [(idx_to_word.get(i, '?'), float(sims[i])) for i in top_idx]

#Donne un dict si besoin
def build_freq_dict(similar_words):
    # similar_words = [("brilliant", 0.92), ("stunning", 0.88)...]
    return {word: score for word, score in similar_words}