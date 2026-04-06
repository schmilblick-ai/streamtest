import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_outlier_WV(model, words):
    try:
        return model.wv.doesnt_match(words)
    except KeyError as e:
        return None

def outlier_scores_WV(model, words):
    results = []
    for w in words:
        others = [x for x in words if x != w]
        try:
            sims = [model.wv.similarity(w, o) for o in others]
            results.append((w, sum(sims)/len(sims)))
        except KeyError:
            pass
    return sorted(results, key=lambda x: x[1])

#directement sur la liste des mots et le tokenizer avec les poids -> change l'appel
def outlier_scores(words, tokenizer, weights):
    vecs = {}
    for w in words:
        idx = tokenizer.word_index.get(w)
        if idx and idx < weights.shape[0]:
            vecs[w] = weights[idx]

    scores = []
    for w, vec in vecs.items():
        others = [v for k, v in vecs.items() if k != w]
        if others:
            sims = cosine_similarity([vec], others)[0]
            scores.append((w, float(np.mean(sims))))

    return sorted(scores, key=lambda x: x[1])

def find_outlier(words, tokenizer, weights):
    scores = outlier_scores(words, tokenizer, weights)
    return scores[0][0] if scores else None