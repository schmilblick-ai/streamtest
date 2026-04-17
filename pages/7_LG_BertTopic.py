
import streamlit as st
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from backend.utils import markdown_table_to_df #, clean_markdown, clean_markdown0

#caching INPUT DATA
@st.cache_data
def load_modelLG():
  return np.load(Path("data/dataLG/df40_comments_embeddings_GPU.npy"))

@st.cache_data
def load_df40_phrases_comments():
  import pandas as pd

  df40_phrases = pd.read_csv(Path("data/dataLG/df40_phrases.csv"),sep=";")
  df40_comments = pd.read_csv(Path("data/dataLG/df40_comments.csv"),sep=";")

  #MISERERE when we read back the csv, we have some NaN in the comment and phrase columns, we need to fill them with empty string for the next steps
  df40_comments['commentaire']=df40_comments['commentaire'].fillna("")
  df40_phrases['Phrase']=df40_phrases['Phrase'].fillna("")

  return df40_phrases, df40_comments

#on ne peut mettre embeddings en param car c'est non hashable, 
# mais on peut mettre la fonction de chargement des embeddings en cache et l'appeler dans la fonction qui fait le calcul
# seul problème si on reuse la même fonctin plusieurs fois ?

MODELS = {
    "df40_comments_emb_GPU": {"repo": "data/dataLG/df40_comments_embeddings_GPU.npy", "dim": 384 },
    "df40_comments_emb_CPU": {"repo": "data/dataLG/df40_comments_embeddings_CPU.npy", "dim": 384},
    "df40_phrases_emb_CPU": {"repo": "data/dataLG/df40_phrases_embeddings_CPU.npy", "dim": 384},
    # ajouter ici...
}

@st.cache_data(show_spinner="Calcul t-SNE en cours...")
def compute_tsne(model_id='df40_comments_emb_GPU'):
  #warning tsne is very slow so two components directly - that's make is very limited
  #en passant le model_id qui est hashable, on peut faire du caching même si les embeddings eux-mêmes ne le sont pas  
  embeddings = globals().get(model_id)
  tsne = TSNE(n_components=2, random_state=42)
  return tsne.fit_transform(embeddings)

@st.cache_data
def compute_pca(embeddings):
  pca = PCA(n_components=2)
  return pca.fit_transform(embeddings), pca.explained_variance_ratio_, pca.components_



@st.cache_data
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

# on charge les embeddings syndiqués de ce streamlit uniquement, 
# pour éviter les problèmes de transfert de gros fichiers et pour permettre une mise à jour locale si besoin
# la première fois peut être longue pour les phrases
df40_comments_emb_GPU, df40_comments_emb_CPU, df40_phrases_emb_CPU = load_embeddings()




#loading load the embeddings from the GPU encoding for visualization
df40_comments_emb_GPU = load_modelLG()

# tabs creation
tab1, tab2, tab3 = st.tabs(["☁️ TSNE reduc", "📈 Basic dim reduction", "🤖 géométrie sémantique"])


with tab1:
    
    embeddings_2d = compute_tsne()

    st.header("☁️ TSNE reduc : basic visualisation of embeddings with dimension reduction - via TSNE or PCA")
    st.markdown(""" 
      embeddings est un tableau numpy de dimension (nombre de phrases, dimension de l'embedding) -> 2mn46    <br>
      on peut faire une réduction de dimension pour visualiser les embeddings, par exemple avec t-SNE ou PCA <br>
      TSNE is T-distributed Stochastic Neighbor Embedding
    """)
    
    fig=plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
    plt.title("Visualisation des embeddings des commentaires (GPU)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    #plt.show()
    
    st.plotly_chart(fig, use_container_width=True)

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

    emb_2d, variance_ratio, components = compute_pca(df40_comments_emb_GPU)

    fig_pca=plt.figure(figsize=(10, 10))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
    plt.title("PCA des embeddings des commentaires (GPU)")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")

   # 3. Chart et Statistiques PCA
    col_chart, col_stats = st.columns(2)

    with col_chart:
      with st.expander("Chart PCA", expanded=True):
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_stats:
      with st.expander("PCA Statistiques Variance expliquée par chaque axe", expanded=True):
        st.success(f"""
          **Traitement effectué :**
          - Axe 1 explique : {variance_ratio[0]*100:.1f}%
          - Axe 2 explique : {variance_ratio[1]*100:.1f}%
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
  

  @st.cache_data
  def compute_kmeans(embeddings, N):
    kmeans   = KMeans(n_clusters=N, random_state=42)
    return kmeans.fit_predict(embeddings_2d)

  N=8
  with st.expander("PCA Statistiques Variance expliquée par chaque axe, **Traitement effectué :** - Clustering KMeans avec N={N} clusters", expanded=False):
    st.success  ("""Pour chaque cluster visible dans t-SNE
      - → trouvez les reviews représentatives du cluster
      - → lisez-les 
      - → donnez un nom humain au cluster
    """)

    clusters = compute_kmeans(df40_comments_emb_GPU, N)

    df40_phrases, df40_comments = load_df40_phrases_comments()

    for cluster_id in range(N):
        mask    = clusters == cluster_id
        samples = np.array(df40_comments["commentaire"])[mask][:10]   # 3 reviews représentatives
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
    @st.cache_data
    def compute_pca_tsne(embeddings):
      # Étape 1 — PCA d'abord pour réduire à 50 dims
      # (accélère t-SNE × 10 sans perte significative)
      pca        = PCA(n_components=50, random_state=42)
      emb_pca    = pca.fit_transform(df40_comments_emb_GPU)

      # Étape 2 — t-SNE pour la visualisation finale
      tsne       = TSNE(n_components=2, random_state=42,
                        perplexity=30, max_iter=1000)
      emb_tsne   = tsne.fit_transform(emb_pca)
      return emb_tsne, emb_pca, pca.explained_variance_ratio_, pca.components_

    emb_tsne, emb_pca, variance_ratio, components = compute_pca_tsne(df40_comments_emb_GPU)

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
  
    @st.cache_data
    def compute_umap(embeddings):
      reducer   = umap.UMAP(n_components=2, random_state=42)
      emb_umap  = reducer.fit_transform(embeddings)
      return emb_umap
    
    emb_umap  = compute_umap(df40_comments_emb_CPU)

    # Ce qui compte → ce qu'on projetee sur la carte - pas les axes eux-mêmes
    notes = df40_comments.note
    fig=plt.figure(figsize=(10, 10))
    plt.scatter(
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
    import torch
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

    #version plus simple
    def modele_id_encode_batch(model_id, sentences, batch_size = 32 ):
      # 1. Chargement (inchangé)
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      model = AutoModel.from_pretrained(model_id)
      model.eval() # Mode inférence

      def mean_pooling(model_output, attention_mask):
          token_embeddings = model_output[0]
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
          return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
          with torch.no_grad():
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

    #petite précision pour streamlit
    #normalement, df40_comments_emb_GPU suffit, pas besoin du load
    #model_GPU_complet a été sauvé en temps que df40_comments_emb_GPU

    #AnchorsEmbeddings = encode_batch(AnchorsLst, model_GPU_complet)

    model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    AnchorsEmbeddings = modele_id_encode_batch(model_id,AnchorsLst)

    # c'est la fonction EmbeddAnchors qu'on met en cache
    #AnchorsEmbeddings=EmbeddAnchors(df40_comments_emb_GPU,AnchorsLst)
    #on NE peut PAS s'abstenir de push le PM-MiniLM-L12-v2-complet et faire le lien 
    # entre les deux via le nom de la variable df40_comments_emb_GPU qui est déjà dans le repo, 
    # on ne peut pas éviter d'avoir à gérer un asset supplémentaire qui est lourd et qui peut faire doublon
    # avec l'asset df40_comments_emb_GPU déjà présent dans le repo
    #sinon on ne peut pas faire l'embedding - l'injection, l'incorporation de nouveaux topics d'ancrage 
    # ou de nouvelles reviews à la volée, ce qui est le but de cette section de l'app


    st.write("Embeddings des topics d'ancrage :")
    st.write(AnchorsEmbeddings) 


