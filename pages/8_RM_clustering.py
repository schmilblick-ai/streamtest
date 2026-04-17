### importation des modules
import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
#import plotly.graph_objects as go
#import torch
from transformers import AutoModel #, AutoTokenizer
#from peft import PeftModel
from sentence_transformers import SentenceTransformer, models
#from umap import UMAP
import joblib
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from backend.utils import sync_project_files
import os
#from pathlib import Path
from PIL import Image

#from sklearn.cluster import HBDSCAN
import umap

# On augmente la limite (ou on la désactive, à vos risques et périls)
Image.MAX_IMAGE_PIXELS = None

### fonctions

def assign_cluster_from_knn(
    target_embedding,
    corpus_embeddings,
    labels,
    k=10,
    return_details=False
):
    """
    Assigne un embedding à un cluster basé sur les k plus proches voisins.

    Parameters
    ----------
    target_embedding : np.ndarray de shape (d,) ou (1, d)
    corpus_embeddings : np.ndarray de shape (n_samples, d)
    labels : np.ndarray de shape (n_samples,)
    k : int
        nombre de voisins à considérer
    return_details : bool
        si True, retourne aussi les voisins et leurs labels

    Returns
    -------
    best_cluster : int
    (optionnel) details : dict
    """

    # --- sécurité shape ---
    target_embedding = np.asarray(target_embedding)
    if target_embedding.ndim == 1:
        target_embedding = target_embedding.reshape(1, -1)

    corpus_embeddings = np.asarray(corpus_embeddings)
    labels = np.asarray(labels)

    # --- kNN ---
    k = min(k, len(corpus_embeddings))

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(corpus_embeddings)

    distances, indices = nn.kneighbors(target_embedding)

    neighbor_labels = labels[indices[0]]

    # --- vote majoritaire ---
    label_counts = Counter(neighbor_labels)
    best_cluster = label_counts.most_common(1)[0][0]

    if return_details:
        return best_cluster, {
            "neighbor_labels": neighbor_labels,
            "neighbor_indices": indices[0],
            "distances": distances[0],
            "votes": dict(label_counts)
        }

    return best_cluster
    
main_data = "data/data_final_streamlit"  

@st.cache_data
def reducing_head(pipeline):

  embeddings_paraph,_ = load_embs(pipeline)
  
  reducer = umap.UMAP(
    n_neighbors=5
    , min_dist=0.0
    , metric='cosine'
    , n_components=5                       
    , random_state=42)
  
  reducer.fit_transform(embeddings_paraph)
  reducer.embedding_  = reducer.embedding_.astype(np.float32)
  reducer._raw_data = reducer._raw_data.astype(np.float32)
  reducer.graph_ = None
  return embeddings_paraph, reducer  

#@st.cache_resource
@st.cache_data
def load_models(pipeline):
  
  if pipeline in ['paraphrase_UMAP_HDBSCAN', 'paraphrase_UMAP_KMEANS'] :
    ### loading models
    model_paraphrase = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    #this is not used and overloaded a few lines later or?
    model = AutoModel.from_pretrained(f"{main_data}/final_lora")

  if pipeline in ['paraphrase_LoRA_UMAP_HDBSCAN', 'paraphrase_LoRA_UMAP_KMEANS'] :
    word_embedding_model = models.Transformer(f"{main_data}/final_lora")
    pooling_model = models.Pooling(
      word_embedding_model.get_word_embedding_dimension(),
      pooling_mode_mean_tokens=True
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

  ### loading commonly the reducer 
  _ , pipeline_reducer = reducing_head(pipeline)

  if pipeline == 'paraphrase_UMAP_HDBSCAN' :
    return model_paraphrase, pipeline_reducer
  elif pipeline == 'paraphrase_UMAP_KMEANS' :
    return model_paraphrase, pipeline_reducer
  elif pipeline == 'paraphrase_LoRA_UMAP_HDBSCAN' :
    return model, pipeline_reducer
  elif pipeline == 'paraphrase_LoRA_UMAP_KMEANS' :
    return model, pipeline_reducer
    
@st.cache_data
def load_embs(pipeline):
  ### loading embeddings and labels for clustering of new input
  data_paraphrase_hdbscan = np.load(f"{main_data}/labels_embeddings_paraphrase_hdbscan.npz")
  embeddings_hdbscan = data_paraphrase_hdbscan["embeddings"]
  labels_hdbscan = data_paraphrase_hdbscan["labels"]

  data_paraphrase_kmeans = np.load(f"{main_data}/labels_embeddings_paraphrase_kmeans.npz")
  embeddings_kmeans = data_paraphrase_kmeans["embeddings"]
  labels_kmeans = data_paraphrase_kmeans["labels"]

  data_paraphrase_lora_hdbscan = np.load(f"{main_data}/labels_embeddings_paraphrase_lora_hdbscan.npz")
  embeddings_lora_hdbscan = data_paraphrase_lora_hdbscan["embeddings"]
  labels_lora_hdbscan = data_paraphrase_lora_hdbscan["labels"]

  data_paraphrase_lora_kmeans = np.load(f"{main_data}/labels_embeddings_paraphrase_lora_kmeans.npz")
  embeddings_lora_kmeans = data_paraphrase_lora_kmeans["embeddings"]
  labels_lora_kmeans = data_paraphrase_lora_kmeans["labels"]
  
  if pipeline == 'paraphrase_UMAP_HDBSCAN' :
    return embeddings_hdbscan, labels_hdbscan
  elif pipeline == 'paraphrase_UMAP_KMEANS' :
    return embeddings_kmeans, labels_kmeans
  elif pipeline == 'paraphrase_LoRA_UMAP_HDBSCAN' :
    return embeddings_lora_hdbscan, labels_lora_hdbscan
  elif pipeline == 'paraphrase_LoRA_UMAP_KMEANS' :
    return embeddings_lora_kmeans, labels_lora_kmeans

@st.cache_data
def load_embs_3d(pipeline):
  ### loading embeddings and labels for clusters visualization
  data_paraphrase_hdbscan = np.load(f"{main_data}/labels_embeddings_paraphrase_hdbscan_3d.npz")
  embeddings_hdbscan = data_paraphrase_hdbscan["embeddings"]
  labels_hdbscan = data_paraphrase_hdbscan["labels"]

  data_paraphrase_kmeans = np.load(f"{main_data}/labels_embeddings_paraphrase_kmeans_3d.npz")
  embeddings_kmeans = data_paraphrase_kmeans["embeddings"]
  labels_kmeans = data_paraphrase_kmeans["labels"]

  data_paraphrase_lora_hdbscan = np.load(f"{main_data}/labels_embeddings_paraphrase_lora_hdbscan_3d.npz")
  embeddings_lora_hdbscan = data_paraphrase_lora_hdbscan["embeddings"]
  labels_lora_hdbscan = data_paraphrase_lora_hdbscan["labels"]

  data_paraphrase_lora_kmeans = np.load(f"{main_data}/labels_embeddings_paraphrase_lora_kmeans_3d.npz")
  embeddings_lora_kmeans = data_paraphrase_lora_kmeans["embeddings"]
  labels_lora_kmeans = data_paraphrase_lora_kmeans["labels"]
  
  if pipeline == 'paraphrase_UMAP_HDBSCAN' :
    return embeddings_hdbscan, labels_hdbscan
  elif pipeline == 'paraphrase_UMAP_KMEANS' :
    return embeddings_kmeans, labels_kmeans
  elif pipeline == 'paraphrase_LoRA_UMAP_HDBSCAN' :
    return embeddings_lora_hdbscan, labels_lora_hdbscan
  elif pipeline == 'paraphrase_LoRA_UMAP_KMEANS' :
    return embeddings_lora_kmeans, labels_lora_kmeans


@st.cache_data
def predict(text, pipeline):
  ### to assign cluster id to new input
    model, umap = load_models(pipeline)

    emb = model.encode([text])  # (1, d)
    emb_umap = umap.transform(emb)  # (1, d_reduced)

    if pipeline in ['paraphrase_UMAP_HDBSCAN', 'paraphrase_LoRA_UMAP_HDBSCAN']:
        emb_corpus, labels = load_embs(pipeline)
        return assign_cluster_from_knn(emb_umap, emb_corpus, labels)

    elif pipeline == 'paraphrase_UMAP_KMEANS':
        kmeans = joblib.load(f"{main_data}/kmeans_paraphrase.pkl")
        return kmeans.predict(emb_umap)[0]

    elif pipeline == 'paraphrase_LoRA_UMAP_KMEANS':
        kmeans = joblib.load(f"{main_data}/kmeans_paraphrase_lora.pkl")
        return kmeans.predict(emb_umap)[0]




### Première partie: Introduction, exploration des données
st.title("Projet de clustering des avis utilisateurs TrustPilot")
#st.sidebar.title("Sommaire")
pages=["Experimental setup", "Modélisation"]
#page=st.sidebar.radio("Aller vers", pages)

tab7, tab8 = st.tabs([f"☁️ {pages[0]}", f"📈 {pages[1]}"])  

### un bouton pour controler les load de data

if st.button("Load data"):
  Proj="data_final_streamlit"
  with st.spinner(f"Récupération des ressources pour {Proj} depuis Hugging Face... [{Proj}_Path]"):
    
    #"Variable globale {Proj}_Path créée avec le chemin local synchronisé"
    globals()[f"{Proj}_Path"] = sync_project_files(Proj, repo_id="Robin-la-Lune/streamTest") 
    st.success(f"ressources pour {Proj} chargés !")

if os.path.exists(f"{main_data}/40k_langdetect.csv") and os.path.exists(f"{main_data}/40k_final_process.csv"):
  st.success("Données déjà présentes localement, pas besoin de les recharger depuis Hugging Face !")
    
  with tab7 : 
    
    ### Chargement des données :
    df_lang = pd.read_csv(f'{main_data}/40k_langdetect.csv', sep=',', header=0, index_col=0) # chargement des données traitées pour la langue
    df_final = pd.read_csv(f'{main_data}/40k_final_process.csv', sep=',', header=0, index_col=0) # chargement des données après preprocessing final
    
    st.write("### DataVizualization and experimental setup")

  ### Figure interactive : langues détectées dans le corpus d'avis
    fig = px.histogram(
      df_lang,
      x="lang",
      color="lang",
      color_discrete_sequence=px.colors.qualitative.Bold,
      opacity=0.7,
      title="Langues détectées dans le corpus d'avis",
      labels={
            "lang": "langues détectées",
            "count": "nombre d'avis",
          },
      
      
      )
    fig.update_layout(
    yaxis_title="Nombre d'avis"
    )
    fig.update_traces(
    hovertemplate="Valeur: %{x}<br>Count: %{y}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)
    
  ### Figure interactive : distribution des avis en fonction des notes et de leur taille
    df_count = df_final['note'].value_counts().sort_index().reset_index()
    df_count.columns = ['note', 'count']

    fig1 = px.bar(df_count, x='note', y='count')

    fig2 = px.histogram(df_final, x='length_comm')

    df_grouped = df_final[['note','length_comm']].groupby('note').median().reset_index()

    fig3 = px.bar(df_grouped, x='note', y='length_comm')

    fig = make_subplots(
      rows=1, cols=3,
      subplot_titles=[
        "Distribution des notes",
        "Distribution longueur commentaires",
        "Longueur médiane vs note"
      ]
    )
    fig.update_layout(template="plotly_dark")
    fig.update_layout(title="Distribution des avis en fonction de leur taille et des notes associées")
    for trace in fig1.data:
      fig.add_trace(trace, row=1, col=1)

    for trace in fig2.data:
      fig.add_trace(trace, row=1, col=2)

    for trace in fig3.data:
      fig.add_trace(trace, row=1, col=3)

    st.plotly_chart(fig, use_container_width=True)
    
  ### figure statique: experimental setup
    st.image(f"{main_data}/experimental_procedure.png")

  with tab8: 
    
    st.write("### Clustering")
    
    
    choix = ['paraphrase_UMAP_HDBSCAN', 'paraphrase_UMAP_KMEANS', 'paraphrase_LoRA_UMAP_HDBSCAN', 'paraphrase_LoRA_UMAP_KMEANS']
    option = st.selectbox('Choix du pipeline', choix)
    
    if option == 'paraphrase_UMAP_HDBSCAN':
      
      st.write(""" * Silouhette score après optimisation UMAP + clustering: **0.63**
            \n* Silouhette score après post-process (agglomerative clustering + cosine distance): **-0.07**
          """)
      
      df_clust = pd.DataFrame( {  'Cluster ID':[0,1,2,3,4,5,6,7,8,9,-1],
                    'Nom du Cluster':['Satisfaction globale très positive sans détail spécifique', 'Insatisfaction majeure liée aux délais de livraison', 'Bon rapport qualité/prix avec livraison rapide', 'Problèmes liés à la livraison et à l’emballage des pièces', 'Conformité du produit et respect des attentes', 'Expérience centrée sur Oscaro, pièces et logistique', 'Fidélité client et satisfaction durable', 'Prix compétitifs et attractifs (cluster marginal)', 'Satisfaction liée au conseil et au service téléphonique', 'Expérience sans problème (RAS et conforme)', 'Bruit non clusturisé'],
                    'Nombre d\'avis':[2448, 7873, 16728, 5439, 1979, 2991, 208, 12, 319, 213, 51]
                    })
      df_clust.set_index('Cluster ID', inplace=True)
      st.write(df_clust)

      embeddings, labels = load_embs_3d(option)
      labels = labels.astype(str)
      fig = px.scatter_3d(
      x=embeddings[:, 0],
      y=embeddings[:, 1],
      z=embeddings[:, 2],
      color=labels
      
      )
      st.plotly_chart(fig)
      
      user_input = st.text_input("Tape ton avis ici")
      if user_input:
        cluster = predict(user_input, option)
        st.write("Cluster:", cluster)

      #Best UMAP: (5, 0.0, 5) score: 0.5811600089073181
      #Best HDBSCAN min_cluster_size: 10 score: 0.6324224472045898

    
    elif option == 'paraphrase_UMAP_KMEANS':

      st.write(""" * Silouhette score après optimisation UMAP + clustering: **0.44**
          """)
      df_clust = pd.DataFrame( {  'Cluster ID':[0,1,2,3,4,5,6,7,8,9,-1],
                    'Nom du Cluster':['Insatisfaction forte liée aux délais de livraison', 'Satisfaction globale simple et sans critique', 'Conformité produit avec respect des délais', 'Achat de pièces automobiles (Oscaro et batterie)', 'Livraison rapide et produit conforme', 'Avis mitigé avec déception ponctuelle logistique', 'Problèmes de colis ouvert et emballage défectueux', 'Recherche de pièces compatibles avec véhicule', 'Service efficace et sans reproche', 'Bon rapport qualité/prix des produits', 'Bruit non clusturisé'],
                    'Nombre d\'avis':[5623, 2702, 6532, 2723, 5541, 197, 6750, 1657, 114, 6422, 0]
                    })
      df_clust.set_index('Cluster ID', inplace=True)
      st.write(df_clust)
          
      embeddings, labels = load_embs_3d(option)
      labels = labels.astype(str)
      fig = px.scatter_3d(
      x=embeddings[:, 0],
      y=embeddings[:, 1],
      z=embeddings[:, 2],
      color=labels
      )
      st.plotly_chart(fig)
      
      user_input = st.text_input("Tape ton avis ici")
      if user_input:
        cluster = predict(user_input, option)
        st.write("Cluster:", cluster)
    
    elif option == 'paraphrase_LoRA_UMAP_HDBSCAN':
    
      st.write(""" * Silouhette score après optimisation UMAP + clustering: **0.95**
            \n* Silouhette score après post-process (agglomerative clustering + cosine distance): **-0.02**
          """)
      
      df_clust = pd.DataFrame( {  'Cluster ID':[0,1,2,3,4,5,6,7,8,9,-1],
                    'Nom du Cluster':['Retours centrés sur colis et expérience Oscaro', 'Expérience globale autour des pièces, prix et livraison', 'Problématiques liées aux colis et aux pièces', 'Satisfaction globale produit et prix', 'Service rapide et efficace avec forte satisfaction', 'Retours spécifiques sur expérience site et commande', 'Satisfaction élevée avec rapidité et conformité', 'Bon rapport qualité/prix des pièces', 'Conformité des produits et satisfaction générale', 'Très forte satisfaction et réactivité perçue', 'Bruit non clusturisé'],
                    'Clusters TrustPilot':['order', 'delivery service', 'mistake', 'price', 'service', 'site', 'warranty', 'product', 'quality', 'customer service', 'None'],
                    'Nombre d\'avis':[1247, 8310, 3646, 9807, 1948, 70, 1064, 7020, 5093, 56, 0]
                    })
      df_clust.set_index('Cluster ID', inplace=True)
      st.write(df_clust)
      
      embeddings, labels = load_embs_3d(option)
      labels = labels.astype(str)
      fig = px.scatter_3d(
      x=embeddings[:, 0],
      y=embeddings[:, 1],
      z=embeddings[:, 2],
      color=labels
      )
      st.plotly_chart(fig)
      
      user_input = st.text_input("Tape ton avis ici")
      if user_input:
        cluster = predict(user_input, option)
        st.write("Cluster:", cluster)
    
    elif option == 'paraphrase_LoRA_UMAP_KMEANS':
      
      st.write(""" * Silouhette score après optimisation UMAP + clustering: **0.30**
          """)
          
      df_clust = pd.DataFrame( {  'Cluster ID':[0,1,2,3,4,5,6,7,8,9,-1],
                    'Nom du Cluster':['Expérience centrée sur Oscaro et gestion des colis', 'Service efficace et livraison rapide', 'Achat de pièces avec bon rapport qualité/prix', 'Produit conforme avec bon rapport qualité/prix', 'Bon rapport qualité/prix des pièces détachées', 'Satisfaction globale avec livraison rapide', 'Bon rapport qualité/prix des pièces', 'Conformité produit avec bon rapport qualité/prix', 'Satisfaction liée au colis et aux pièces reçues', 'Satisfaction globale sur produits et prix', 'Bruit non clusturisé'],
                    'Nombre d\'avis':[4180, 3305, 3794, 3130, 3472, 5291, 4494, 3259, 3879, 3457, 0]
                    })
      df_clust.set_index('Cluster ID', inplace=True)
      st.write(df_clust)
      
      embeddings, labels = load_embs_3d(option)
      labels = labels.astype(str)
      fig = px.scatter_3d(
      x=embeddings[:, 0],
      y=embeddings[:, 1],
      z=embeddings[:, 2],
      color=labels
      )
      st.plotly_chart(fig)

      user_input = st.text_input("Tape ton avis ici")
      if user_input:
        cluster = predict(user_input, option)
        st.write("Cluster:", cluster)
