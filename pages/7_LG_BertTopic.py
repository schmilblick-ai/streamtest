
import os
import umap
import json
import joblib
import numpy as np
import pandas as pd
#from turtle import pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from backend.utils import markdown_table_to_df, Step, Cache_Disk #, clean_markdown, clean_markdown0

from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

#caching INPUT DATA
@st.cache_data
def load_modelLG():
  return np.load(Path("data/dataLG/df40_comments_embeddings_GPU.npy"))

#loader les trois est trop gros pour le cache, on peut faire un cache par chargement et les appeler dans la fonction qui fait le calcul
#faisons un par un

@st.cache_resource
def load_embeddings(model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'): 

  #reprise sur panne des embeddings des phrases 
  #loading des embeddings syndiqués de ce streamlit uniquement
  specific="data/dataLG/df40_phrases_embeddings_CPU.npy"
  if not (Path(specific).exists()):
      st.warning("Embeddings des phrases reviews non trouvés(taille de transfert trop importante, recréation en local)")
      st.warning("Veuillez patienter, encodage en cours... (cela peut prendre plusieurs minutes (~ 12mn))")
      df40_phrases, _ = load_df40_phrases_comments()

      #phrases_embeddings = encode_batch(phrases_list, model_GPU_complet)
      those_embeddings = modele_id_encode_batch(model_id,df40_phrases['Phrase'].tolist())
      np.save(Path(specific), those_embeddings)
      st.success(f"🤏Encodage {specific} terminé et sauvegardé !")
  else:
      st.success(f"👍Encodage {specific} déjà existant !")

  return (np.load(Path("data/dataLG/df40_comments_embeddings_GPU.npy"))
    , np.load(Path("data/dataLG/df40_comments_embeddings_CPU.npy"))
    , np.load(Path("data/dataLG/df40_phrases_embeddings_CPU.npy"))
    )


#la aussi etre plus spécifique plutot que tout ouvrir
#penser plus globalement chaque traitement, et séparé les backends des frontends par taille
@st.cache_data
def load_df40_comments():
  #MISERERE when we read back the csv, we have some NaN in the comment and phrase columns, we need to fill them with empty string for the next steps
  df40_comments = pd.read_csv(Path("data/dataLG/df40_comments.csv"),sep=";")
  df40_comments['commentaire']=df40_comments['commentaire'].fillna("")
  return df40_comments

@st.cache_data
def load_df40_phrases():
  #MISERERE when we read back the csv, we have some NaN in the comment and phrase columns, we need to fill them with empty string for the next steps
  df40_phrases = pd.read_csv(Path("data/dataLG/df40_phrases.csv"),sep=";")
  df40_phrases['Phrase']=df40_phrases['Phrase'].fillna("")
  return df40_phrases


@st.cache_data
def load_df40_phrases_comments():
  return load_df40_phrases, load_df40_comments

#on ne peut mettre embeddings en param car c'est non hashable, 
# mais on peut mettre la fonction de chargement des embeddings en cache et l'appeler dans la fonction qui fait le calcul
# seul problème si on reuse la même fonctin plusieurs fois ?

MODELS = {
    "df40_comments_emb_GPU": {"repo": "data/dataLG/df40_comments_embeddings_GPU.npy", "dim": 384 },
    "df40_comments_emb_CPU": {"repo": "data/dataLG/df40_comments_embeddings_CPU.npy", "dim": 384},
    "df40_phrases_emb_CPU": {"repo": "data/dataLG/df40_phrases_embeddings_CPU.npy", "dim": 384},
    # ajouter ici...
}

@st.cache_resource(show_spinner="Calcul t-SNE en cours...")
def compute_tsne00(model_id='df40_comments_emb_GPU', N=2):
  #warning tsne is very slow so two components directly - that's make is very limited
  #en passant le model_id qui est hashable, on peut faire du caching même si les embeddings eux-mêmes ne le sont pas  
  embeddings = globals().get(model_id)
  tsne = TSNE(n_components=2, random_state=42)
  return tsne.fit_transform(embeddings)

#définition d'un test de validité entre deux fichiers
def is_cache_valid(source_path, cache_path):
    """Vérifie si le cache est présent et plus récent que la source."""
    if not os.path.exists(cache_path):
        return False
    # Comparaison des dates de modification
    source_mtime = os.path.getmtime(source_path)
    cache_mtime = os.path.getmtime(cache_path)
    return cache_mtime > source_mtime

#on le cache pas en mémoire, on le cache sur disque pour éviter les problèmes de mémoire et de temps de calcul
#on donne le src_file pour gérer la validité du cache, et le tgt_file pour stocker le résultat du calcul
#on donne l'embeddings
@Step("Calcul du t-SNE avec cache disque")
def compute_tsne01(embeddings,src_file,tgt_file = "tsne_cache_40k.joblib"):
  if is_cache_valid(src_file, tgt_file):
    # On charge les 40k x 3 coordonnées directement
    print("Chargement des coordonnées t-SNE depuis le disque...")
    return joblib.load(tgt_file)
  else:
    st.write("☕Calcul du t-SNE (Préparez un café, c'est long)...")
    # On passe de 384 dims à 3 dims
    tsne = TSNE(n_components=3, init='pca',          # Initialisation PCA : accélère la convergence
      learning_rate='auto', # Évite que l'algorithme ne tourne en rond
      method='barnes_hut',  # L'algorithme optimisé pour les grands datasets
      n_jobs=-1            # Utilise tous les coeurs de ton processeur
    )
    coords = tsne.fit_transform(embeddings)
    # On sauvegarde l'array [40000, 3]
    joblib.dump(coords, tgt_file)
    return coords

# On introduit un cache disk pour chainer les fichiers  fait un métaprogramation python - un décorateur
# On doit juste définir src_file et tgt_file dans la définition de la fonction et le décorateur s'occupe du reste

@Cache_Disk()
@Step("Calcul du t-SNE avec cache disque")
def compute_tsne(embeddings,src_file,tgt_file = "tsne_cache_40k.joblib"):
  tsne = TSNE(n_components=3, init='pca',          # Initialisation PCA : accélère la convergence
    learning_rate='auto', # Évite que l'algorithme ne tourne en rond
    method='barnes_hut',  # L'algorithme optimisé pour les grands datasets
    n_jobs=-1            # Utilise tous les coeurs de ton processeur
  )
  coords = tsne.fit_transform(embeddings)
  return coords

@st.cache_resource
def compute_pca00(model_id='df40_comments_emb_GPU'):
  embeddings = globals().get(model_id)
  pca = PCA(n_components=3)
  return pca.fit_transform(embeddings), pca.explained_variance_ratio_, pca.components_

@Cache_Disk()
@Step("Calcul du PCA avec cache disque")
def compute_pca(embeddings,src_file="",tgt_file=""):
  pca = PCA(n_components=3)
  return pca.fit_transform(embeddings), pca.explained_variance_ratio_, pca.components_



# on charge les embeddings syndiqués de ce streamlit uniquement, 
# pour éviter les problèmes de transfert de gros fichiers et pour permettre une mise à jour locale si besoin
# la première fois peut être longue pour les phrases
df40_comments_emb_GPU, df40_comments_emb_CPU, df40_phrases_emb_CPU = load_embeddings()

#loading load the embeddings from the GPU encoding for visualization
df40_comments_emb_GPU = load_modelLG()


# tabs creation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["☁️ TSNE reduc"
    , "📈 Reduction & Clustering"
    , "🤖 géométrie sémantique"
    , "🤹 Classification Multicat"
    , "static views"
    , "dynamic views"
    ])

def plot_3d_sampling(df,xyzc=['tsne_x','tsne_y','tsne_z','cluster'],sldKey="p3d"):
    thisContainer=st.container()

    col1, col2= thisContainer.columns([0.2,0.8])
    #st.divider()
    col1.subheader(f"🎮 Contrôles du Graphique {sldKey}")
    
    # 1. Slider pour l'échantillonnage (préserve la fluidité)
    n_points = col1.slider(
        "Nombre de points à afficher", 
        min_value=1000, 
        max_value=len(df), 
        value=min(10000, len(df)),
        step=1000,
        key=f"{sldKey}_1"
    )
    
    # 2. Slider pour l'opacité (aide à voir la densité)
    opacity_val = col1.slider("Opacité", 0.1, 1.0, 0.5,key=f"{sldKey}_2")

    # Échantillonnage aléatoire
    df_sample = df.sample(n=n_points, random_state=42)

    # Création du plot
    fig = px.scatter_3d(
        df_sample, 
        x=xyzc[0], 
        y=xyzc[1], 
        z=xyzc[2],
        color=xyzc[3], # Doit être une chaîne de caractères ou catégorie
        opacity=opacity_val,
        title=f"Projection 3D ({n_points} points)",
        color_continuous_scale='Viridis', # Ou un autre schéma de couleur
        hover_data=['text_preview'] if 'text_preview' in df.columns else None
    )

    # Réglages fins des marqueurs
    fig.update_traces(marker=dict(size=2))
    
    # Amélioration de l'ergonomie (taille du graphe)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='Axe X',
            yaxis_title='Axe Y',
            zaxis_title='Axe Z'
        )
    )
    col2.plotly_chart(fig, use_container_width=True)
    #st.plotly_chart(fig_pca, use_container_width=True)

    return thisContainer

with tab1:
    
    st.header("☁️ TSNE reduc : basic visualisation of embeddings with dimension reduction - via TSNE or PCA")
    st.markdown(""" 
      embeddings est un tableau numpy de dimension (nombre de phrases, dimension de l'embedding) -> 2mn46    <br>
      on peut faire une réduction de dimension pour visualiser les embeddings, par exemple avec t-SNE ou PCA <br>
      TSNE is T-distributed Stochastic Neighbor Embedding
    """)

    #très long sauf si en cache disque
    embeddings_3d = compute_tsne(df40_comments_emb_GPU
        , src_file="data/dataLG/df40_comments_embeddings_GPU.npy", tgt_file="data/dataLG/tsne_cache_40k.joblib")
    
    df3d=pd.DataFrame(embeddings_3d, columns=['tsne_x', 'tsne_y', 'tsne_z'])
    df3d['cluster']=1
    
    st.write(df3d.head())

    figTsne=plot_3d_sampling(df3d,sldKey="tsne3d")
    


    st.subheader("comparaison TSNE et une Principal Component Analysis")

    raw_text="""Rappel:\n
        Comparaison
        | TYPE                     | PCA                                      | t-SNE                      | 
        | ------------------------ | ---------------------------------------- | -------------------------  | 
        | ***Type***               | linéaire                                 | non-linéaire               | 
        | ***Objectif***           | variance maximale                        | voisinage local            | 
        | ***Vitesse***            | ⚡ très rapide                           | 🐢 lent sur gros datasets | 
        | ***Reproductible***      | ✅ toujours                              | ⚠️ dépend du random_state | 
        | ***Dimensions cibles***  | n quelconque                              | 2 ou 3 seulement          |
        | ***Distances globales*** | ✅ préservées                            | ❌ non fiables            | 
        | ***Clusters visuels***   | ⚠️ parfois invisibles                    | ✅ très lisibles          | 
        | ***Interprétable***      | ✅ axes = combinaisons de features       | ❌ axes sans sens         | 
        | ***Nouveaux points***    | ✅ transform() direct                    | ❌ doit tout recalculer   | 

    """

    df = markdown_table_to_df(raw_text)
    st.dataframe(df, use_container_width=False)
    #st.write( clean_markdown(raw_text))
    #st.markdown(clean_markdown(raw_text))
    
    st.subheader("PCA : Principal Component Analysis")

    #emb_2d, variance_ratio, components = compute_pca00("df40_comments_emb_GPU")
    emb_3d, variance_ratio, components = compute_pca(df40_comments_emb_GPU
          , src_file="data/dataLG/df40_comments_embeddings_GPU.npy", tgt_file="data/dataLG/pca_cache_40k.joblib")

    if False:
      fig_pca=plt.figure(figsize=(10, 10))
      plt.scatter(emb_3d[:, 0], emb_3d[:, 1], alpha=0.5)
      plt.title("PCA des embeddings des commentaires (GPU)")
      plt.xlabel("Composante principale 1")
      plt.ylabel("Composante principale 2")
      plt.ylabel("Composante principale 3")

    dfpca_3d=pd.DataFrame(emb_3d, columns=['pca_x', 'pca_y', 'pca_z'])
    st.write(dfpca_3d.head())
    #simulacre de clustering - et oui le graphique attend un cluster
    dfpca_3d['cluster']=1
    
    
   # 3. Chart et Statistiques PCA
    col_chart, col_stats = st.columns(2)

    with col_chart:
      with st.expander("Chart PCA", expanded=True):
        plot_3d_sampling(dfpca_3d,xyzc=['pca_x', 'pca_y', 'pca_z', 'cluster']  ,sldKey='pca3d')
        

    with col_stats:
      with st.expander("PCA Statistiques Variance expliquée par chaque axe", expanded=True):
        st.success(f"""
          **Traitement effectué :**
          - Axe 1 explique : {variance_ratio[0]*100:.1f}%
          - Axe 2 explique : {variance_ratio[1]*100:.1f}%
          - Axe 3 explique : {variance_ratio[2]*100:.1f}%
          - Total          : {sum(variance_ratio)*100:.1f}%
          """)
        st.success(f"""
          **Les composantes — quelles dimensions du modèle pèsent le plus**
          - Composante 1 top dims : {np.argsort(np.abs(components[0]))[-5:]}
          """)    
 

with tab2:
  st.header("🤖 Modélisation : basic clustering of embeddings with KMeans or DBSCAN")
  st.markdown(""" 
    embeddings est un tableau numpy de dimension (nombre de phrases, dimension de l'embedding) -> 2mn46    <br>
    on peut faire du clustering sur les embeddings, par exemple avec KMeans ou DBSCAN <br>
  """)     


  from sklearn.cluster import KMeans
  
  @st.cache_resource
  def compute_kmeans00(model_id='df40_comments_emb_GPU', N=8):
    embeddings_2d = globals().get(model_id)
    kmeans   = KMeans(n_clusters=N, random_state=42)
    return kmeans.fit_predict(embeddings_2d)

  @Cache_Disk()
  @Step("Compute Kmeans for clustering on 40k")
  def compute_kmeans(embeddings_2d, N=8,src_file="",tgt_file=""):
    kmeans   = KMeans(n_clusters=N, random_state=42)
    return kmeans.fit_predict(embeddings_2d)
  
  NbCLusters=8
  with st.expander("Clustering KMeans avec N={NbCLusters} clusters puis PCA Statistiques Variance expliquée par chaque axe, **Traitement effectué :** -", expanded=False):
    st.success  ("""Pour chaque cluster visible dans df40K
      - → trouvez les reviews représentatives du cluster
      - → lisez-les 
      - → donnez un nom humain au cluster
    """)

    #bon on ajoute une première clusterisation gratuite
    clusters = compute_kmeans(df40_comments_emb_GPU, NbCLusters
      , src_file="data/dataLG/df40_comments_embeddings_GPU.npy", tgt_file="data/dataLG/kmeans_cache_40k.joblib")

    df40_comments = load_df40_comments()
    
    for cluster_id in range(NbCLusters):
        mask    = clusters == cluster_id
        samples = np.array(df40_comments["commentaire"])[mask][:10]   # N reviews représentatives
        # on pass le nom du cluster et le nombre de reviews dans le cluster pour mieux comprendre la nature du cluster
        st.write(f"\nCluster {cluster_id} ({mask.sum()} reviews) :")
        for r in samples:
            st.write(f"  → {r[:160]}")

  with st.expander("Combinaison des deux ! PCA suivie de t-SNE (-> 2 mn 19)", expanded=False):
    st.success  ("""
    Pour chaque cluster visible dans t-SNE
      - → trouvez les reviews représentatives du cluster
      - → lisez-les 
      - → donnez un nom humain au cluster
    """)


    # embeddings shape → (n_reviews, 384)
    @st.cache_resource
    def compute_pca_tsne00(model_id='df40_comments_emb_GPU'):
      embeddings = globals().get(model_id)
      # Étape 1 — PCA d'abord pour réduire à 50 dims
      # (accélère t-SNE × 10 sans perte significative)
      pca        = PCA(n_components=50, random_state=42)
      emb_pca    = pca.fit_transform(embeddings)

      # Étape 2 — t-SNE pour la visualisation finale
      tsne       = TSNE(n_components=2, random_state=42,
                        perplexity=30, max_iter=1000)
      emb_tsne   = tsne.fit_transform(emb_pca)

      return emb_tsne, emb_pca, pca.explained_variance_ratio_, pca.components_

    # embeddings shape → (n_reviews, 384)
    @Cache_Disk()
    def compute_pca_tsne(embeddings,src_file="", tgt_file=""):
      # Étape 1 — PCA d'abord pour réduire à 50 dims
      # (accélère t-SNE × 10 sans perte significative)
      pca        = PCA(n_components=50, random_state=42)
      emb_pca    = pca.fit_transform(embeddings)

      # Étape 2 — t-SNE pour la visualisation finale
      tsne       = TSNE(n_components=2, random_state=42,
                        perplexity=30, max_iter=1000)
      emb_tsne   = tsne.fit_transform(emb_pca)

      return emb_tsne, emb_pca, pca.explained_variance_ratio_, pca.components_
    
    #question, est-ce que joblib dump des tuples et est-ce que le load renvoi le tuple d'origine ? OUI

    emb_tsne, emb_pca, variance_ratio, components = compute_pca_tsne(df40_comments_emb_GPU
      , src_file="data/dataLG/df40_comments_embeddings_GPU.npy", tgt_file="data/dataLG/tsnepca_cache_40k.joblib")

    perplexity = min(30, max(5, int(np.sqrt(len(emb_tsne)))))

    st.success  (f"""
          Pour chaque cluster visible dans t-SNE
          - PCA  : {df40_comments_emb_GPU.shape} → {emb_pca.shape}
          - t-SNE: {emb_pca.shape}   → {emb_tsne.shape} 
          - Variance expliquée par les 50 composantes PCA : {variance_ratio.sum()*100:.1f}%
          - Composantes PCA les plus importantes : {np.argsort(np.abs(components[0]))[-5:]}
          - perplexity ≈ nombre de voisins proches considérés
            - règle : entre 5 et 50, souvent sqrt(n_samples)
            - perplexity : {perplexity} (ajusté automatiquement en fonction du nombre de reviews)    
    """)


  #uv add umap-learn --active
  with st.expander("UMAP — similarité sémantique des reviews", expanded=False):
  
    @st.cache_resource
    def compute_umap00(model_id='df40_comments_emb_CPU'):
      embeddings = globals().get(model_id)
      reducer   = umap.UMAP(n_components=2, random_state=42)
      emb_umap  = reducer.fit_transform(embeddings)
      return emb_umap

    @Cache_Disk()
    def compute_umap(embeddings,src_file="", tgt_file=""):
      reducer   = umap.UMAP(n_components=3, random_state=42)
      emb_umap  = reducer.fit_transform(embeddings)
      return emb_umap

    emb_umap  = compute_umap(df40_comments_emb_CPU
      , src_file="data/dataLG/df40_comments_embeddings_GPU.npy", tgt_file="data/dataLG/umap_cache_40k.joblib")
    
    dfumap3d=pd.DataFrame(embeddings_3d, columns=['umap_x', 'umap_y', 'umap_z'])
    #on reprend les clusters de notre kmeans, on pourrait prendre les notes aussi
    dfumap3d['cluster']=clusters

    plot_3d_sampling(dfumap3d,xyzc=['umap_x', 'umap_y', 'umap_z', 'cluster']  ,sldKey='umap3d')
    
    if False:
      # COMMENT PLOT 2D - Ce qui compte → ce qu'on projete sur la carte - pas les axes eux-mêmes
      notes = df40_comments.note
      fig=plt.figure(figsize=(10, 10))
      plt.scatter_3d(
          emb_umap[:, 0], emb_umap[:, 1],
          c=notes,          # note 1 à 5 → couleur
          cmap="RdYlGn",    # rouge=mauvais, vert=bon
          alpha=0.5,
          s=10,
      )
      plt.colorbar(label="Note client")
      plt.title("UMAP — similarité sémantique des reviews") 

      st.plotly_chart(fig, use_container_width=True)

  with st.expander("Vérification qualitative simple sur 20 avis", expanded=False):
    def checkOneOut(emb_umap,IDTGT=4):

      idx_proche = np.argsort(
          np.linalg.norm(emb_umap - emb_umap[IDTGT], axis=1)
      )[:10]
      st.info(f"""
        Review de référence :
        ➡️ {IDTGT} : {df40_comments.commentaire[IDTGT]}
        \nReviews les plus proches en UMAP :
        """)
      for i in idx_proche[1:]:
        st.write(f"  →  ({df40_comments.note[i]}) {i} {df40_comments.commentaire[i][:160]}")

    for k in range(20):
      checkOneOut(emb_umap,k)



with tab3:    
  st.header("🤖 de la géométrie à la sémantique")
  with st.expander("🤖 géométrie sémantique : exploration de la structure des embeddings et de leur interprétabilité", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
      st.subheader("1. Géométrie des embeddings")
      st.markdown("""Les embeddings sont des points dans un espace de haute dimension (ex : 384 dimensions pour les modèles de phrases). 
        La géométrie de cet espace encode la sémantique : les points proches ont des significations similaires, les directions peuvent correspondre à des concepts, etc. 
        En explorant cette géométrie, on peut découvrir des relations intéressantes entre les reviews, identifier des clusters de thèmes, ou même faire du "word arithmetic" (ex : "bon" - "mauvais" ≈ "excellent" - "terrible"). 
      """)
      st.subheader("2. De la géométrie à la sémantique")
      st.markdown("""
        Passer de la géométrie (des points dans un espace) à la sémantique (des mots métier compréhensibles). 
        Une fois que vos phrases sont regroupées par UMAP + Clustering (comme HDBSCAN ou K-Means), on a des groupes, mais ils sont "muets".
        Pour extraire les mentions les plus significatives, l'approche de référence en NLP moderne est le `**c-TF-IDF (Class-based TF-IDF)**`.
        - `TF-IDF` classique : mesure l'importance d'un mot dans un document par rapport à une collection de documents :
        - `c-TF-IDF` : mesure l'importance d'un mot dans un cluster de phrases par rapport à tous les clusters.
                  
        En appliquant c-TF-IDF à vos clusters, vous obtenez les mots les plus caractéristiques de chaque groupe, ce qui vous permet de leur donner un nom métier compréhensible (ex : "problème de livraison", "qualité du produit", etc.).    
      """)

    with col2:
      st.subheader("""3. Next? L'approche par "Centroïde" """)
      st.markdown("""
        Si vous voulez trouver la phrase|metion exacte la plus représentative :
        - Calculez le centroïde de votre cluster (la moyenne de tous les vecteurs du groupe).
        - Cherchez la phrase dont l'embedding est le plus proche de ce centroïde (distance cosinus minimale).

        Cette phrase est votre "médaille d'or 🥇" : c'est elle qui résume le mieux le topic métier.
        Par la suite, on pourrait définir nos mentions et embedder cette mention pour faire du few-shot learning ou du zero-shot learning sur de nouvelles reviews.

        ### 4. Conseils métier pour la catégorisation

        Stop-words personnalisés : En entreprise, certains mots sont partout (ex: le nom de votre boîte). 
        Ajoutez-les à votre liste de stop_words pour qu'ils ne polluent pas vos clusters.

        Les N-grammes : Ne vous limitez pas aux mots seuls. "Livraison" c'est bien, "Livraison domicile" c'est un topic métier.

        Le filtrage POS (Part-of-Speech) : Souvent, seuls les Noms et les Adjectifs portent le sens métier. 
        Vous pouvez utiliser spaCy pour ne garder que ces types de mots avant de faire le c-TF-IDF.
    """)
      
    with col3:
      st.subheader("Idées méthodes")
      st.markdown(""" 
      Méthode "Zéro-Shot" (La plus rapide, sans entraînement)
      Utiliser la Similarité Cosinus par rapport à des "ancres" (des topics d'intérêt).

      Définissez des centres d'intérêt (ex: "Délais", "Emballage", "Service Client", "Prix").
      list of 11 Topics to embed : ['Délais de livraison', 'Emballage', 'Service Client', 'Prix', 'Qualité du produit', 'Facilité d'utilisation', 'Fiabilité', 'Communication', 'Retour et remboursement', 'Expérience globale', 'Autre']
      Astuce, mieux que les mots, une phrase métier complète pour donner de la profondeur à l'ancrage : 
      "Je suis mécontent du délai de livraison" → embedding de cette phrase comme ancre pour le topic "Délais de livraison".
      permet d'envisager le classer les définitions par rapport aux score de revue peut-être intéressant 
      également pour faire du tri dans les définitions métier et ne garder que les plus pertinentes.
      
      Puis -> Computation of the topics ancres embedding pour vectorisé ces définitions.
      Encodez ces définitions-clés avec le même modèle (GPU ou l'importance de raisoné isomodèle ainsi que des stratégies de refresh/ réindexation).

      Calculez la distance entre chaque avis et chaque définition-clé.  
      """)

  with st.expander("Encodage des topics d'ancrage", expanded=False):
  
    #Une deuxième méthod similaire à encode large corpuse bellow, mais affiner
    from openvino import Core
    
    from transformers import AutoTokenizer, AutoModel
    from dataclasses import dataclass
    import numpy as np
    from tqdm import tqdm # Pour voir la barre de progression tqkm = progresser en arabe

    #a KISS for managing the structure of the output model, to allow having several at same time
    @dataclass
    class OVModel:
      compiled_model  : object
      tokenizer       : object
      batch_size      : int
      seq_length      : int
      padding         : object   # "max_length" ou True
      proc_type       : str
      input_names     : list[str]
      output_names    : list[str]

    #same function purpose as the get_ov_model above but with a dataclass output
    def load_model(model_path: str, xmlSrc: str, proc_type: str = "auto", batch_size: int = 16, seq_length: int = 128):
        """
        Charge et compile le modèle OpenVINO selon le procType.
        proc_type : "auto" | "gpu" | "npu" | "cpu"
        """
        core = Core()
        ov_model = core.read_model(f"{model_path}/{xmlSrc}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        #detection dynamique des inputs
        #Le NPU est particulièrement sensible aux inputs manquants — là où CPU/GPU peuvent parfois tolérer un input absent, le NPU crashe le kernel immédiatement.
        existing_inputs = [inp.any_name for inp in ov_model.inputs]

        # ── Résolution auto
        if proc_type == "auto":
            devices = [d.lower() for d in core.available_devices]
            if "npu" in devices:
                proc_type = "npu"
            elif "gpu" in devices:
                proc_type = "gpu"
            else:
                proc_type = "cpu"
            print(f"✓ Auto-détection → {proc_type.upper()}")

        # ── GPU — shape statique après reshape
        if proc_type == "gpu":
            #static declaration will make things fail or crash
            #v_model.reshape({ "input_ids":      [batch_size, seq_length], "attention_mask": [batch_size, seq_length] })
            #trying dynamic
            ov_model.reshape({name: [batch_size, seq_length] for name in existing_inputs})
            compiled_model = core.compile_model(ov_model, "GPU", config={ "CACHE_DIR": "data/gpu_cache", "PERFORMANCE_HINT": "THROUGHPUT" })
            padding      = "max_length"
            print(f"✓ GPU compilé — batch={batch_size}, seq={seq_length}")

        # ── NPU — shape statique obligatoire, batch petit
        elif proc_type == "npu":
            npu_batch = min(batch_size, 16)   # NPU préfère les petits batches
            ov_model.reshape({name: [npu_batch, seq_length] for name in existing_inputs})
            compiled_model = core.compile_model(ov_model, "NPU", config={ "CACHE_DIR": "data/npu_cache", "PERFORMANCE_HINT": "LATENCY" })
            batch_size   = npu_batch
            padding      = "max_length"
            print(f"✓ NPU compilé — batch={npu_batch}, seq={seq_length} (compilation longue au 1er run)")

        # ── CPU — shape dynamique, pas de reshape
        elif proc_type == "cpu":
            compiled_model = core.compile_model( ov_model, "CPU", config={ "CACHE_DIR": "data/cpu_cache", "PERFORMANCE_HINT": "THROUGHPUT" })
            padding      = True   # dynamique — pad au plus long du batch
            print("✓ CPU compilé — shapes dynamiques")

        else:
            raise ValueError(f"proc_type inconnu : '{proc_type}' — utilisez 'gpu', 'npu', 'cpu' ou 'auto'") #pas encore de QPU ou de TPU ou de *PU
        
        #on prépare ici, au lieu de pendant l'usage, la création des deux structures input_names et output_names
        return OVModel(
            compiled_model  = compiled_model
            ,tokenizer      = tokenizer
            ,batch_size     = batch_size
            ,seq_length     = seq_length
            ,padding        = padding
            ,proc_type      = proc_type
            ,input_names    = [inp.any_name for inp in compiled_model.inputs]
            ,output_names   = [out.any_name for out in compiled_model.outputs]
        )

    def encode_batch(texts: list[str], m: OVModel) -> np.ndarray:
      """Encode une liste de textes en embeddings."""
      if not texts:
          return np.zeros((0, 384), dtype=np.float32)
      
      # ── Inputs attendus par ce modèle compilé sont retrouvé dans le dataclass passé en param    
      all_embeddings = []

      for i in range(0, len(texts), m.batch_size):
          batch = texts[i:i + m.batch_size]

          # Complète le batch si incomplet (obligatoire pour GPU/NPU statiques)
          padding_needed = m.batch_size - len(batch)
          batch_padded   = batch + [""] * padding_needed

          inputs = m.tokenizer( batch_padded, padding=m.padding, truncation=True, max_length=m.seq_length,  return_tensors="np"
              , return_token_type_ids=True,  # toujours demandé, ignoré si absent
              )
          
          #dynamo model inputs
          model_inputs = { name: inputs[name] for name in m.input_names if name in inputs }
          #print(model_inputs)
          outputs = m.compiled_model(model_inputs)

          #mean pooling, ici suivant les formes vectories on peut avoir des token_vectors ou token_states, ou bien des embeddings
          raw_outputs = outputs[m.output_names[0]]  #was = next(iter(outputs.values())) that takes the first layer but is less robusts

          # ── Détection automatique selon le shape de sortie
          if raw_outputs.ndim == 3:
              # Modèle complet → (batch, seq, dim) → mean pooling nécessaire
              mask       = inputs["attention_mask"][:, :, np.newaxis].astype(np.float32)
              embeddings = (raw_outputs * mask).sum(axis=1) / mask.sum(axis=1)
          elif raw_outputs.ndim == 2:
              # Modèle fp16 optimum-cli → (batch, dim) → déjà poolé
              embeddings = raw_outputs
          else:
              raise ValueError(f"Shape de sortie inattendu : {raw_outputs.shape}")

          # Retire les embeddings de padding
          all_embeddings.append(embeddings[:len(batch)])

          if i % 1000 == 0:
              print(f"Progression encode_batch : {i}/{len(texts)} avis traités...")

      return np.vstack(all_embeddings)

    model_GPU_complet = load_model(
        model_path = Path("data/dataLG/PM-MiniLM-L12-v2-complet"),
        xmlSrc = Path("openvino/openvino_model.xml"),
        #model_path = "data/PM-MiniLM-L12-v2-fp16",    xmlSrc = "openvino_model.xml",
        proc_type  = "gpu",   # "gpu" | "npu" | "cpu" | "auto" 
        batch_size = 64,
        seq_length = 512,
    )
    st.write("""👍`model_GPU_complet` pret à l'emploi""")

    #version plus simple
    def modele_id_encode_batch(model_id, sentences, batch_size = 32 ):
      from torch import sum as torch_sum
      from torch import clamp as torch_clamp
      from torch import no_grad as torch_no_grad
      # 1. Chargement (inchangé)
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      model = AutoModel.from_pretrained(model_id)
      model.eval() # Mode inférence
      def mean_pooling(model_output, attention_mask):
          token_embeddings = model_output[0]
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
          return torch_sum(token_embeddings * input_mask_expanded, 1) / torch_clamp(input_mask_expanded.sum(1), min=1e-9)

      # 2. Paramètres de traitement
      # sentences 
      # Ajustez selon votre RAM (32 est très sûr)

      all_embeddings = []

      # 3. Boucle de traitement par batchs
      for i in tqdm(range(0, len(sentences), batch_size)):
          # Extraction du batch
          batch_sentences = sentences[i : i + batch_size]
          
          # Tokenization du batch uniquement
          encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
          
          # Inférence sans calcul de gradient (gain mémoire énorme)
          with torch_no_grad():
              model_output = model(**encoded_input)
          
          # Pooling
          batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
          
          # Stockage en CPU / Numpy pour libérer la mémoire vive de travail
          all_embeddings.append(batch_embeddings.cpu().numpy())

      # 4. Concaténation finale
      sentence_embeddings = np.vstack(all_embeddings)

      print(f"Structure finale : {sentence_embeddings.shape}") # Devrait être (80000, 384)
      return sentence_embeddings


    def PlotCompareEmbeding(emb0,lab0,emb1,lab1,Title="Sample Embeddings"):
        from sklearn.metrics.pairwise import cosine_similarity

        fig, ax = plt.subplots(figsize=(16, 6))
        plt.plot(emb0, label=lab0)
        plt.plot(emb1, label=lab1)
        plt.xlabel(f"{emb0.shape[0]} Dimensions")
        plt.ylabel("Embedding Value")
        plt.title(Title + f" (Cosine Similarity: {cosine_similarity(emb0.reshape(1, -1), emb1.reshape(1, -1))[0][0]:.4f})")
        plt.legend()
        plt.show()

    AnchorsStr=st.text_input("Entrez une liste de topics séparés par des virgules : "
      , value="livraison, delivery service,order, commande,product,price,prix,quality,service,mistake,warranty,website,refund"
      , key="topics_input")
    
    AnchorsLst=[Anchor.strip() for Anchor in AnchorsStr.split(",")]
    st.write(f"Topics d'ancrage : {AnchorsLst}")
    df_Anchors=pd.DataFrame(AnchorsLst, index=AnchorsLst, columns=["label"])

    st.write("""👍`AnchorsLst` et `df_Anchors` pret à l'emploi""")

    #petite précision pour streamlit
    #normalement, df40_comments_emb_GPU suffit, pas besoin du load
    #model_GPU_complet a été sauvé en temps que df40_comments_emb_GPU

    #AnchorsEmbeddings = encode_batch(AnchorsLst, model_GPU_complet)

    model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    AnchorsEmbeddings = modele_id_encode_batch(model_id,AnchorsLst)
    
    st.write("Embeddings des topics d'ancrage :")
    st.write(AnchorsEmbeddings) 

    st.write(f"""👍`model_id` ({model_id}) et `AnchorsEmbeddings` pret à l'emploi""")

    st.write("""
     c'est la fonction EmbeddAnchors qu'on met en cache
    
    AnchorsEmbeddings=EmbeddAnchors(df40_comments_emb_GPU,AnchorsLst)
    on NE peut PAS s'abstenir de push le PM-MiniLM-L12-v2-complet et faire le lien 

     entre les deux via le nom de la variable df40_comments_emb_GPU qui est déjà dans le repo, 

     on ne peut pas éviter d'avoir à gérer un asset supplémentaire qui est lourd et qui peut faire doublon
             
     avec l'asset df40_comments_emb_GPU déjà présent dans le repo
             
    sinon on ne peut pas faire l'embedding - l'injection, l'incorporation de nouveaux topics d'ancrage 
             
     ou de nouvelles reviews à la volée, ce qui est le but de cette section de l'app
    """)


    
  with st.expander("Encodage des 'Définition' de topics d'ancrage - synthétisé par IAGen", expanded=False):

    # A l'aide d'une IAgen, on lui donne une liste d'idée de mentions et on lui demande de les regrouper et de les définir
    # définition de Themes mentions et emmeding, sur la base d'une première réfléxion sur le sujet
    # une approche suivante consistera à trouver des mentions importantes nouvelles dans les data - découvrir des trends
    # souvent on passe à coté des choses importantes

    #Autre points, classer les phrases, et non les avis complet, pour du multitopic
    #dernier points, par avis ou phrase, donner les plus haut, pas just le plus haut -> multi topic 

    with open('themes.json', 'r', encoding='utf-8') as f:
      themes_data = json.load(f)

    # Créer la map de couleurs pour Plotly
    mention_color_map = {v['label']: v['color'] for k, v in themes_data.items()}
    if False:
      # Utiliser dans Plotly Express
      fig = px.scatter(df, x="x", y="y", color="Theme_Label", color_discrete_map=mention_color_map)

    # Transformer en DataFrame pour l'affichage ou le processing
    df_themes = pd.DataFrame.from_dict(themes_data, orient='index')

    # Extraire la liste pour l'embedding
    df_themes["WholeDef"]=df_themes["label"]+" : "+df_themes["keywords"].astype(str) +" : "+df_themes["definition"]
    WholeDefs_to_embed = df_themes['WholeDef'].tolist()
    
    st.write(df_themes,WholeDefs_to_embed)

    model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    WholeDefinitionsEmbeddings = modele_id_encode_batch(model_id,WholeDefs_to_embed)
    st.write(f"""👍`model_id` ({model_id}) et `WholeDefinitionsEmbeddings` pret à l'emploi""")

    st.write("Embeddings des 7 Définitions métiers Ancrage :")
    st.write(WholeDefinitionsEmbeddings) 

    # Représentation graphique comparé avec deux sliders sur les mentions vs les définitions
    # très sympas à voir


  def show_embedding_comparison(topics1_names, topics1_embeddings,topics2_names, topics2_embeddings):
      st.header("📊 Comparaison des Signatures Sémantiques")
      
      # Création du conteneur avec ratio 1/4 - 3/4
      col1, col2 = st.columns([0.2, 0.8])
      
      with col1:
          # Dans un formulaire nommé 'comparateur'
          with st.form("my_comparison_form"):
            st.subheader("Configuration")
            # Sliders Combo pour choisir les deux thèmes
            topic1_name = st.selectbox("Topic A (Bleu)" , topics1_names, index=0)
            topic2_name = st.selectbox("Topic B (Rouge)", topics2_names, index=0)
            
            # Récupération des index
            #st.write(topic1_name,topics1_names,topic2_name,topics2_names)
            idx1 = np.where(topics1_names['label'] == topic1_name)[0][0] #topics1_names.index()
            idx2 = np.where(topics2_names['label'] == topic2_name)[0][0] #topics2_names.loc[topic2_name]

            # bouton magique obligatoire dans un formulaire
            submitted = st.form_submit_button("Lancer la comparaison")
      if not submitted:
        with col2:
            st.info("Sélectionnez deux thèmes et cliquez sur le bouton pour comparer.")
      else:      
        with col2:    
            # --- PRÉPARATION DU DF_SAMPLE ---
            # On crée un DataFrame "long" pour Plotly
            emb1 = topics1_embeddings[idx1]
            emb2 = topics2_embeddings[idx2]
            
            # Création des index de dimensions (0 à 767 par exemple)
            dims = np.arange(len(emb1))
            
            df1 = pd.DataFrame({'Dimension': dims, 'Valeur': emb1, 'Topic': topic1_name})
            df2 = pd.DataFrame({'Dimension': dims, 'Valeur': emb2, 'Topic': topic2_name})
            #df_sample = pd.concat([df1, df2])

            # --- CALCUL DE LA SIMILARITÉ ---
            # On reshape car sklearn attend un tableau 2D
            v1 = np.array(emb1).reshape(1, -1)
            v2 = np.array(emb2).reshape(1, -1)
            score = cosine_similarity(v1, v2)[0][0]

            # Affichage d'un petit indicateur visuel dans la col1
            st.metric("Similarité Cosinus", f"{score:.4f}")
            
            if score > 0.85:
                st.success("Sémantique très proche")
            elif score < 0.50:
                st.warning("Sémantique éloignée")    
            
        
            # --- CRÉATION DU CHART PLOTLY ---
            fig = go.Figure()

            # Courbe Topic 1
            fig.add_trace(go.Scatter(
                x=df1['Dimension'], y=df1['Valeur'],
                mode='lines', name=topic1_name,
                line=dict(color='#1f77b4', width=2)
            ))

            # Courbe Topic 2
            fig.add_trace(go.Scatter(
                x=df2['Dimension'], y=df2['Valeur'],
                mode='lines', name=topic2_name,
                line=dict(color='#d62728', width=2, dash='dot')
            ))

            fig.update_layout(
                title=f"Comparaison : {topic1_name} vs {topic2_name}",
                xaxis_title="Dimensions du vecteur d'embedding",
                yaxis_title="Amplitude",
                hovermode="x unified",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

  show_embedding_comparison(df_Anchors, AnchorsEmbeddings, df_themes, WholeDefinitionsEmbeddings)      

  def show_similarity_heatmap(names_A, embeds_A, names_B, embeds_B):
    # 1. Calcul de la matrice n x p
    # cosine_similarity renvoie directement une matrice si on lui donne deux listes
    matrix = cosine_similarity(embeds_A, embeds_B)
    #matrix_rounded=np.round(matrix,2)

    # 2. Création du graphique
    fig = px.imshow(
        matrix,
        x=names_B, # Labels en haut
        y=names_A, # Labels à gauche
        color_continuous_scale='RdBu_r', # Bleu (froid/loin) à Rouge (chaud/proche)
        aspect="auto",
        title="Matrice de Similarité Cosinus : Ensemble A vs Ensemble B",
        labels=dict(color="Similarité"),
        #zmin=0, zmax=1 # Optionnel : fixe l'échelle entre 0 et 1 sinon entre min et max ce qui est plus accentué en couleur
    )
    
    if False:
      # Ajout des scores numériques dans les cases (si pas trop de données)
      if len(names_A) * len(names_B) < 1000:
          fig.update_traces(text=np.round(matrix, 2), texttemplate="%{text}")

      fig.update_xaxes(side="top") # Plus facile à lire en haut

    # --- FORMATAGE DES DÉCIMALES DANS LES CASES ---
    fig.update_traces(
        #text=matrix_rounded, 
        #texttemplate="%{text:.2f}", # Force l'affichage de 2 chiffres après la virgule
        text=np.round(matrix, 2),
        texttemplate="<b>%{text:.2f}</b>", # <b> pour le GRAS
        textfont=dict(
            family="Arial",
            size=30,           # Taille des chiffres
            color="#0e1117" #"black"      # Couleur des chiffres (ex: "white", "black", "yellow")
        ),        
        hovertemplate="A: %{y}<br>B: %{x}<br>Score: %{z:.2f}<extra></extra>"
    )

    # --- AJUSTEMENT DE LA HAUTEUR ET DU STYLE ---
    fig.update_layout(
        height=800,       # Augmente la hauteur en pixels (ex: 800 au lieu de 450 par défaut)
        width=800,        # None permet de s'adapter à la largeur du container Streamlit
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=50, r=50, t=100, b=50) # Ajuste les marges pour ne pas couper les labels
    )

    # --- STYLE DES LABELS (AXES) ---
    label_style = dict(family="Arial", size=12, color="lightblue") # Style commun

    fig.update_xaxes(
        side="top", 
        tickangle=+10,
        tickfont=label_style # Applique la taille/couleur aux labels X
    )

    st.plotly_chart(fig, use_container_width=True)

  show_similarity_heatmap(df_themes["label"], WholeDefinitionsEmbeddings, df_Anchors.label, AnchorsEmbeddings)    

with tab4:
  st.write("""
    Au salon Big Data e& IA, un présentateur me parlait de leur problématique de classification multilabel
    A l'époque il ne s'avait pas bien faire ...
            
    Quel approche pour nous aujourd'hui,      
    ## 1. Méthode "Zéro-Shot" (La plus rapide, sans entraînement)
    Utiliser la Similarité Cosinus par rapport à des "ancres" (des topics d'intérêt).

    Définissez des centres d'intérêt (ex: "Délais", "Emballage", "Service Client", "Prix").
    list of 11 Topics to embed

    Computation of the topics embedding

    Encodez ces mots-clés avec le même modèle (GPU).

    Calculez la distance entre chaque avis et chaque mot-clé.

    """)

  if False:
    def topMentions(df_emb, df_emb_Ancres,lstTopics ):
      # 1. Les centres d'intérêt (Ancres)
      st.write(df_emb_Ancres.shape) # (11, 384)

      # 2. Calcul des scores pour un avis (index i)
      # avis_vector: [1, 384], topic_vectors: [4, 384]
      for i in np.range(len(df_emb)):   #index de l'avis à analyser
          #on calcule pour chaque point la distance à tous les centres, c'est ici, ce produit cartésien exploratoire
          scores = cosine_similarity(df_emb[i].reshape(1, -1), df_emb_Ancres)[0]

          # 3. Seuil de décision (Threshold)
          # Si le score > 0.4, l'avis appartient au topic
          assigned_topics = [lstTopics[j] for j, score in enumerate(scores) if score > 0.25]
          d=dict(zip(lstTopics, scores))
          
          print(
              f"Avis {i}: {df40_comments['commentaire'].iloc[i]}"
            ,f"Scores : {sorted(d.items(), key=lambda x: x[1], reverse=True)[:5]}" # Affiche les 5 topics les plus proches avec leurs scores
            ,f"Topics assignés : {assigned_topics}"
            ,"\n"
          , sep="\n"
          )
      #illustrations
      #fournir un jeu consistant df_emb, def_emb_Ancres, et lsttopics
      lstTopics=["livraison", "delivery service","order", "commande","product","price","prix","quality","service","mistake","warranty","website","refund"]
      topMentions

  st.write("""            
    Il utilise les embeddings pour créer des clusters, mais il permet à un document d'avoir une distribution de probabilités sur plusieurs clusters.

    Réduction de dimension (UMAP) : les 384 dimensions sont compressées en 5 ou 10 dimensions clés.

    Clustering (HDBSCAN) : Détecte les groupes naturels d'avis.

    Multi-topic : En utilisant l'option calculate_probabilities=True, on obtient pour chaque avis un score pour chaque cluster détecté.

    uv add bertopic umap-learn hdbscan --active
                        
  """)  

  if True:
    st.divider()
    st.write("""### Définition des models support UMAP et HBDSCAN et representation KeyBERT""")
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    # 1. Préparation de la réduction de dimension (UMAP)
    # On réduit de 384 à 5 dimensions pour aider le clustering
    #deux optims en CPU (impossible de tourner ailleurs pour l'umap (en tout cas pas intel)) low_memory et n_jobs
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42, low_memory=False, n_jobs=-1)

    # 2. Préparation du clustering (HDBSCAN)
    # min_cluster_size: taille mini d'un sujet (ex: 50 avis) - ajoutons core_dist_n_jobs
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom'
    , prediction_data=True, core_dist_n_jobs=-1)

    # 3. ── Représentation sémantique — insensible à la langue
    representation = KeyBERTInspired()
    st.write("""👍`umap_model`, `hdbscan_model` et `representation` KeyBERTInspired ready to go""")

  if True:  
    def ReadableProbs(topic_model,probs):
      # On crée une matrice de probabilités lisible
      prob_df = pd.DataFrame(probs)
      #display(prob_df)
      # On cherche les avis qui dépassent 0.15 de probabilité sur au moins 2 topics
      multi_topic_mask = (prob_df > 0.15).sum(axis=1) >= 2
      indices_multi = prob_df.index[multi_topic_mask].tolist()

      st.write(f"Nombre d'avis multi-thématiques détectés : {len(indices_multi)}")

      # Voir un exemple concret d'avis multi-topic
      if indices_multi:
        idx = indices_multi[0]
        st.write(f"\nAvis n°{idx} : {df40_comments.loc[idx].commentaire}")
        # On affiche les thèmes associés
        top_indices = prob_df.iloc[idx].nlargest(3)
        for t_idx, p in top_indices.items():
          if p > 0.1:
            st.write(f" -> Topic {t_idx} (Confiance: {p:.2f}): {topic_model.get_topic(t_idx)[:3]}")     
    st.write("""👍def Matrice de proba lisible `ReadableProbs(topic_model,probs):`""")


    def bench_probs(probs):
      def benchmark_thresholds(probs_matrix):
        probs_array = np.array(probs_matrix)
        for t in np.arange(0, 0.20, 0.01):
          # 1. On crée un masque booléen (True là où c'est > t)
          mask = probs_array > t
          
          # 2. On compte combien de True par ligne (axis=1)
          counts_per_row = np.sum(mask, axis=1)
          
          # 3. On compte combien de lignes ont un score > 1
          count = np.sum(counts_per_row > 1)
          
          perc = (count / len(probs_array)) * 100
          print(f"Seuil {t:.2f} : {count} avis multi-labels ({perc:.1f}%)")

      benchmark_thresholds(probs)
      st.write(probs)
    st.write("""👍 def Préparation Benchmark `Thresholds bench_probs(probs):`""")

  if True:
    st.divider()
    st.write("""Préparation Vectorize
          ## Topic Model avec les stops words
          Avant de voir les centroides, voyons l'incorporation dans le berttopic de la problématique des stops words en anglais et en francais.

          ### les 🛑 stops words multi lingue ( fr, en, custom)
    """)
    
    #on va combiner les deux listes de stop words fr et encode_batch
    from sklearn.feature_extraction.text import CountVectorizer
    import spacy
    import nltk
    from nltk.corpus import stopwords

    # Stopwords français
    nlp_fr       = spacy.load("fr_core_news_sm")
    stops_fr     = nlp_fr.Defaults.stop_words

    # Stopwords anglais
    nltk.download("stopwords", quiet=True)
    stops_en     = set(stopwords.words("english"))

    # Stopwords custom — mots trop génériques pour votre corpus
    stops_custom = {
        "oscaro", "produit", "commande", "livraison",   # FR trop génériques
        "product", "order", "delivery", "item",          # EN trop génériques
    }

    # Union des trois ET CONSTRUCTION DU VECTORIZER
    stops_all = stops_fr | stops_en | stops_custom

    vectorizer = CountVectorizer(
        stop_words  = list(stops_all),
        min_df      = 2,
        ngram_range = (1, 2),
    )
    st.write("""👍 `vectorizer` et stop words `stops_all` en place""")


  if True:
    st.divider()
    st.write("""## La clusterisation, regroupement de nos avis
              ### Définition de théme ANCRE pour orienter le modèle
              #AVOIR avec un wrapper ?
              """)    
    from bertopic.backend import BaseEmbedder

    # On définit des thèmes "ancres" pour guider le modèle
    seed_topic_list = [
        ["livraison", "transporteur", "colis", "paquet"],
        ["délais", "retard", "attente", "rapide", "temps","temporel"],
        ["prix", "cher", "coût", "frais", "argent"]
    ]

    #bon c'est joli, mais il faut un wrapper:
    #push de cette classe dans la zone main
    class OVEmbedder(BaseEmbedder):
        """Wrapper OpenVINO pour BERTopic."""

        def __init__(self, ov_model):
            super().__init__()
            self.ov = ov_model   # la dataclass OVModel

        def embed_documents(self, documents: list[str], verbose: bool = False) -> np.ndarray:
            return encode_batch(documents, self.ov)

        def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
            return self.embed_documents(documents, verbose)

    # Instanciation
    embedder = OVEmbedder(model_GPU_complet)   # ov = notre  OVModel dataclass model_GPU_complet 

    #_p prefix will be for pretrained model definition
    topic_model2_p = BERTopic(embedding_model=embedder, # ici on reprend le model via le wrapper embedder
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model     = vectorizer,  # stops FR + EN ✅
      representation_model = representation,
      language="multilingual", # Important pour le prétraitement (stopwords, etc.)
      calculate_probabilities=True, # CRUCIAL pour le multi-topic
      verbose=True,
      nr_topics=40,
      seed_topic_list = seed_topic_list,  # ← ici on ajoute le seed topic pour l'embeddings ci-dessous
      min_topic_size =  max(5, df40_comments.commentaire.count() // 100)
    )

    st.write("""👍 `topic_model2_p` prétrained BERTopic empilé et pre à l'emploi""")

  if True:  
    # Relancez le fit avec les vecteurs GPU
    if "df40_comments_emb_GPU" not in globals():
        df40_comments_emb_GPU = load_modelLG()
    
    #@Cache_Disk()    
    #def BuildTopicProbs(topic_model, df, emb,src_file="",tgt_file=""):    

    @st.cache_resource
    def BuildTopicProbs(topic_model_name,df,emb):    
        topic_model= globals()[topic_model_name]
        topic, probs = topic_model.fit_transform(df.commentaire, embeddings=emb)
        return topic, probs, topic_model

    #strategy : to go from topic_model2_p to topic_model2 and serialize the trained model
    #Tactict  : to reset a file and restart, touch this particular file to an earlier project date
    #touch -t 197202150000.00 data/dataLG/topProb2_cache_40k.joblib  
    topics2, probs2, topic_model2= BuildTopicProbs("topic_model2_p",df40_comments,df40_comments_emb_GPU)
      #, src_file="data/dataLG/df40_comments_embeddings_GPU.npy", tgt_file="data/dataLG/topProb2_cache_40k.joblib")
    
    st.write("""👍 `topics2`, `probs2` et `topic_model2` BERTopic construits et calculés ou récupés""")

  if True:
    def visusDocs(topic_model):
      # Run the visualization with the original embeddings
      topic_model.visualize_documents(df40_comments.commentaire, embeddings=df40_comments_emb_GPU)
      # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
      reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(df40_comments_emb_GPU)
      topic_model.visualize_barchart()
      topic_model.visualize_topics()
      topic_model.visualize_documents(df40_comments.commentaire, embeddings=df40_comments_emb_GPU)
      topic_model.visualize_documents(df40_comments.commentaire, reduced_embeddings=reduced_embeddings)

    def visuDocAndTopics(topic_model):
        # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(df40_comments_emb_GPU)
        topic_model.visualize_documents(df40_comments.commentaire, reduced_embeddings=reduced_embeddings)

    st.write("""👍 fonction `visusDocs(topic_model):` et `visuDocAndTopics(topic_model)` en place !""")
    
    visuDocAndTopics(topic_model2)

with tab5:
  if True:
    st.divider()
    st.write("### 📈 Visualisation des topics à travers le temps")

    @st.cache_data
    def visucrosstime(topic_model_name,topiclist=[17,29,19, 32,8,2]):
      topic_model1 = globals()[topic_model_name]
      timestamps = pd.to_datetime(df40_comments['date_experience'], errors='coerce').to_list()
      topics_over_time = topic_model1.topics_over_time(df40_comments.commentaire, timestamps,nr_bins=26)

      return topic_model1.visualize_topics_over_time(topics_over_time, topics=topiclist)

    fig_visucrosstime = visucrosstime("topic_model2",topiclist=[17,29,19, 32,8,2])
    st.plotly_chart(fig_visucrosstime, use_container_width=True)

  if True:
    st.write("Topics per Class")
    topics_per_class = topic_model2.topics_per_class(df40_comments.commentaire, classes=df40_comments.note)
    st.plotly_chart(topic_model2.visualize_topics_per_class(topics_per_class), use_container_width=True)

  if True:
    st.write("""### topic distributions on a token-level""")
    # Calculate the topic distributions on a token-level
    topic_distr, topic_token_distr = topic_model2.approximate_distribution(df40_comments.commentaire, calculate_tokens=True)

    # Patch rapide — ajoute applymap comme alias de map - Pour une raison stupide de conflit avec Pandas et Berttopic
    if not hasattr(pd.io.formats.style.Styler, "applymap"):
        pd.io.formats.style.Styler.applymap = pd.io.formats.style.Styler.map

    # Visualize the token-level distributions
    df = topic_model2.visualize_approximate_distribution(df40_comments.commentaire.iloc[4], topic_token_distr[4])
    st.write(df)

with tab4:
  if True:
    st.write("""### Ajoutez la langue comme métadonnée""")
    
    from langdetect import detect, LangDetectException
    
    def detect_language(text: str, default: str = "unknown") -> str:
        """
        Détecte la langue d'un texte.
        Retourne default si le texte est trop court ou vide.
        """
        # Nettoyage minimal
        cleaned = text.strip() if text else ""

        # Guard — minimum de caractères alphabétiques
        alpha_chars = sum(c.isalpha() for c in cleaned)
        if alpha_chars < 10:
            return default

        try:
            return detect(cleaned)
        except LangDetectException:
            return default
    def apply_detect_language(topics1):
      df40_langue = pd.DataFrame({
          "review"  : df40_comments.commentaire,
          "topic"   : topics1,
          "langue"  : [ detect_language(r) for r in df40_comments.commentaire ],
      })

      # Distribution des langues par topic
      st.write(df40_langue.groupby(["topic", "langue"]).size().unstack(fill_value=0))
      #Vous verrez si certains topics sont mono-langue (problème spécifique FR ou EN) ou bilingues (problème universel).
    
    apply_detect_language(topics2)
    st.divider()

    st.write("""
      Résumé
      Composant | Multilingue ? | Action
      ----------| --------------| ----------
      Embeddings MiniLM | ✅ natif | rien à faire
      UMAP + HDBSCAN | ✅ travaille sur vecteurs | rien à faire
      CountVectorizer | ❌ monolingue | union stops FR + EN
      KeyBERTInspired | ✅ sémantique | recommandé
      Métadonnées langue | — | langdetect

    """)
    if True:
      hierarchical_topics = topic_model2.hierarchical_topics(df40_comments.commentaire)
      fig_hierarch=topic_model2.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
      st.plotly_chart(fig_hierarch, use_container_width=True)

      tree = topic_model2.get_topic_tree(hierarchical_topics)
      #TOTALEMENT INADAPTE st.write(tree)
      print(tree)

with tab6:
  if True:
    st.write("""
    définitions des centroid: attention partie réalisé par genAI - disclaimer - pas eu encore le temps de valider complétement
      """)
    
    if False:
      centroides = {
          # ── Négatif
          "DÉLAI":            [1, 3, 14, 38],
          "CONFORMITÉ":       [4, 6, 15, 35, 37],
          "EMBALLAGE":        [5, 21, 33],
          "PRODUIT_ERRONÉ":   [17, 29, 34],
          "BATTERIE":         [19],
          "ESSUIE_GLACE":     [13, 27],

          # ── Positif
          "LIVRAISON_RAPIDE": [7, 9, 12, 26, 31],
          "QUALITÉ_PRIX":     [2, 23, 32],
          "SATISFACTION":     [8, 10, 16, 18, 24, 30],
          "EFFICACITÉ":       [11, 26, 36],
          "CONFORME":         [20, 22, 25],
          "SERVICE_TEL":      [28],
      }

    centroides = {
        # ── NÉGATIF (Problèmes, Délais, SAV)
        "RETARD_LIVRAISON":    [5, 10, 35],      # Trop long, jours retard, envoyé/reçu mail (souvent lié au retard)
        "LITIGE_REMBOURSEMENT": [2, 23],         # Remboursé, remboursement, geste, faire renvoyer
        "EMBALLAGE_DÉGRADÉ":    [8, 28, 34],     # Carton déchiré, colis carton, mondial relay (casier/locker souvent cités en cas de souci)
        "ERREUR_COMMANDES":     [32, 33, 38],    # Bougies (erreur produit), article manquant, non conforme photo
        "PIÈCES_VÉHICULE":      [4, 6],          # Problèmes immatriculation, compatibilité pièces auto
        "ESSUIE_GLACE_LAVE":    [13, 18],        # Spécifique : lave glace, filtres huile (souvent des erreurs de réf)

        # ── POSITIF (Satisfaction, Rapidité, Qualité)
        "LIVRAISON_RAPIDE":     [9, 12, 15, 17], # Envoi rapide, efficace rapide, reçu rapidement
        "RAPPORT_QUALITÉ_PRIX": [3, 19],         # Qualité prix, prix corrects, prix bien
        "SATISFACTION_GLOBALE": [11, 14, 20],    # Parfait bien, satisfait, bien arrivé, problème rien
        "CONFORMITÉ_ATTENTES":  [0, 16, 24, 27], # Article conforme, conforme attentes, correct ras
        "EXPÉRIENCE_SITE":      [7, 25],         # Site facile, pratique, bien recommande
        "SERVICE_CLIENT_TEL":   [21, 26, 30, 31, 36, 37] # Batterie (souvent conseil), service rapide, tel bien, pro
    }
    st.write("👍 `centroides` defined")

    
  if True:
      st.write("""### Multilabélisation des revues""")
      #on définit la fonction de multilabeling basé sur les centroids
      def multilabel_review(topic_id: int, proba: np.ndarray,
                            centroides: dict, threshold: float = 0.15) -> list[str]:
          """
          Retourne les labels d'une review selon son topic principal
          et ses probabilités sur les autres topics.
          """
          labels = []

          for centroide, topics in centroides.items():
              # Topic principal dans ce centroïde
              if topic_id in topics:
                  labels.append(centroide)
                  continue

              # Topics secondaires via probabilités
              proba_centroide = sum(proba[t] for t in topics if t < len(proba))
              if proba_centroide >= threshold:
                  labels.append(centroide)

          return labels if labels else ["AUTRE"]

      df40_comments["labels"] = [ multilabel_review(topic, proba, centroides)  for topic, proba in zip(topics1, probs1 ) ]

      #basic sentiment
      df40_comments["sentiment"] = df40_comments["labels"].apply(
          lambda labels: "NÉGATIF"
          if any(l in ["RETARD_LIVRAISON", "LITIGE_REMBOURSEMENT", "EMBALLAGE_DÉGRADÉ", "ERREUR_COMMANDES", "PIÈCES_VÉHICULE", "ESSUIE_GLACE_LAVE"]
                for l in labels)
          else "POSITIF"
      )

      df40_comments.head()     

  if False:
    df40_comments.labels.value_counts()
    df40_comments.groupby(df40_comments.sentiment + " " + df40_comments.labels.apply(lambda x: (" ").join(x))).count()#["commentaire"]
     

  if False:
    hierarchical_topics = topic_model2.hierarchical_topics(df40_comments.commentaire)
    topic_model2.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(df40_comments_emb_GPU)
    topic_model2.visualize_documents(df40_comments.commentaire, reduced_embeddings=reduced_embeddings)  

    topics_per_class2 = topic_model2.topics_per_class(df40_comments.commentaire, classes=df40_comments.note)
    topic_model2.visualize_topics_per_class(topics_per_class2)

  if False:

    def plot_wordclouds_by_label(
        df          : pd.DataFrame,
        text_col    : str  = "commentaire",
        labels_col  : str  = "labels",
        stopwords   : set  = None,
        max_words   : int  = 70,
        cols        : int  = 3,
        figsize_w   : int  = 20,
    ):
        """
        Projette un wordcloud par label issu de multilabel_review.
        
        df          : DataFrame avec les reviews et leurs labels
        text_col    : colonne contenant le texte des reviews
        labels_col  : colonne contenant les listes de labels
        max_words   : nombre max de mots par wordcloud
        stopwords   : les stopwords déjà défini
        cols        : nombre de colonnes dans la grille
        figsize_w   : largeur totale de la figure
        """

        # ── Récupère tous les labels uniques
        all_labels = sorted(set(
            label
            for labels in df[labels_col]
            for label in labels
        ))

        # ── Grille de subplots
        rows    = (len(all_labels) + cols - 1) // cols
        fig     = plt.figure(figsize=(figsize_w, rows * 4))
        gs      = gridspec.GridSpec(rows, cols, figure=fig)
        
        # ── Stopwords
        stop = stopwords or set()
        
        for idx, label in enumerate(all_labels):
            # ── Reviews de ce label
            mask     = df[labels_col].apply(lambda x: label in x)
            texts    = df.loc[mask, text_col].dropna().tolist()
            corpus   = " ".join(texts)

            # ── Wordcloud
            wc = WordCloud(
                width            = 400,
                height           = 300,
                max_words        = max_words,
                background_color = "white",
                stopwords        = stop,
                max_font_size    = 50,
                colormap         = "RdYlGn" if label in [
                    "LIVRAISON_RAPIDE", "QUALITÉ_PRIX",
                    "SATISFACTION", "EFFICACITÉ",
                    "CONFORME", "SERVICE_TEL",
                ] else "Reds",
                collocations     = False,   # évite les doublons bigrammes
            ).generate(corpus)

            # ── Subplot
            row, col = divmod(idx, cols)
            ax       = fig.add_subplot(gs[row, col])
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(
                f"{label}  ({mask.sum()} reviews)",
                fontsize = 12,
                fontweight = "bold",
                pad      = 10,
            )
            ax.axis("off")

        # ── Subplots vides
        for idx in range(len(all_labels), rows * cols):
            row, col = divmod(idx, cols)
            fig.add_subplot(gs[row, col]).axis("off")

        plt.suptitle(
            "Wordclouds par label — reviews multilabel",
            fontsize = 16,
            fontweight = "bold",
            y        = 1.02,
        )
        plt.tight_layout()
        plt.show()


    # ── Usage
    plot_wordclouds_by_label(
        df         = df40_comments,
        text_col   = "commentaire",
        labels_col = "labels",
        stopwords  = stops_all,
        max_words  = 50,
        cols       = 3,
    )

  if False:
    def plot_wordclouds_by_sentiment(
        df           : pd.DataFrame,
        text_col     : str = "commentaire",
        sentiment_col: str = "sentiment",
        stopwords   : set  = None,
      ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colormaps = {"POSITIF": "Greens", "NÉGATIF": "Reds"}

        # ── Stopwords
        stop = stopwords or set()

        for ax, sentiment in zip(axes, ["POSITIF", "NÉGATIF"]):
            mask   = df[sentiment_col] == sentiment
            corpus = " ".join(df.loc[mask, text_col].dropna())

            wc = WordCloud(
                width            = 600,
                height           = 400,
                max_words        = 160,
                max_font_size    = 50,
                background_color = "white",
                stopwords        = stop,
                colormap         = colormaps[sentiment],
                collocations     = False,
            ).generate(corpus)

            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(
                f"{sentiment}  ({mask.sum()} reviews)",
                fontsize   = 14,
                fontweight = "bold",
            )
            ax.axis("off")

        plt.suptitle("Wordclouds par sentiment", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()


    # ── Usage
    plot_wordclouds_by_sentiment(df40_comments, stopwords=stops_all)