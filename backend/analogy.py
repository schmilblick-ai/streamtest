def compute_analogy_WV(model, pos1, pos2, neg1, topn=5):
    try:
        return model.wv.most_similar(
            positive=[pos1, pos2],
            negative=[neg1],
            topn=topn
        )
    except KeyError:
        return []
    
def compute_analogy(word_a, word_b, word_c, tokenizer, weights, topn=5):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    def vec(w):
        idx = tokenizer.word_index.get(w)
        return weights[idx] if idx else None
    
    try:
        va, vb, vc = vec(word_a), vec(word_b), vec(word_c)
        if any(v is None for v in [va, vb, vc]):
            return []

        target = va - vb + vc
        sims = cosine_similarity([target], weights)[0]
        top_idx = np.argsort(sims)[::-1][:topn+3]

        idx_to_word = {v: k for k, v in tokenizer.word_index.items()}
        exclude = {word_a, word_b, word_c}
        results = [(idx_to_word[i], float(sims[i]))
                for i in top_idx
                if idx_to_word.get(i) not in exclude]
        return results[:topn]
    
    except KeyError:
        return []