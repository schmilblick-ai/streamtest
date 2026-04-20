# backend/utils.py
import re
import os
import gc
import time
import psutil
import dill
import joblib
import inspect
import textwrap
import pandas as pd
import streamlit as st
#from io import StringIO
import plotly.express as px
from functools import wraps
from datetime import datetime
from huggingface_hub import snapshot_download

def clean_markdown0(text: str) -> str:
    """
    Nettoie un markdown récupéré d'un texte brut :
    - supprime l'indentation parasite
    - strip chaque ligne
    - préserve les blocs de code
    """
    # Supprime l'indentation commune (textwrap.dedent)
    text = textwrap.dedent(text)
    
    # Strip chaque ligne individuellement
    lines = [line.strip() for line in text.splitlines()]
    
    return "\n".join(lines)

def clean_markdown(text: str) -> str:
    lines = []
    for line in textwrap.dedent(text).splitlines():
        stripped = line.strip()
        
        # Ligne de tableau : normalise les espaces internes
        if stripped.startswith("|"):
            # Supprime les espaces excessifs entre les pipes
            stripped = re.sub(r'\s*\|\s*', ' | ', stripped)
            # Recolle les pipes de début/fin
            stripped = stripped.strip(" |")
            stripped = f"| {stripped} |"
        
        lines.append(stripped)
    
    # Supprime les lignes vides multiples
    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def render_clean_markdown(text: str):
    """Nettoie et affiche directement dans Streamlit."""
    st.markdown(clean_markdown(text))

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',
                    unsafe_allow_html=True)
        

def markdown_table_to_df(text: str) -> pd.DataFrame:
    """Extrait la première table markdown d'un texte et la retourne en DataFrame."""
    # Isole les lignes de tableau
    lines = [
        line.strip() 
        for line in textwrap.dedent(text).splitlines() 
        if line.strip().startswith("|")
    ]
    # Supprime la ligne séparateur (| --- | --- |)
    lines = [line for line in lines if not re.match(r'^\|[\s\-\|]+\|$', line)]
    
    # Parse
    rows = [
        [cell.strip().strip("*") for cell in line.strip("|").split("|")]
        for line in lines
    ]
    
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

@st.cache_data
def sync_project_files(subproject_data,repo_id = "schmilblick-ai/streamlearn_data"):
    """
    Télécharge récursivement le dossier du projet depuis HF.
    Se comporte comme un rsync sélectif.
    repo_id = "schmilblick-ai/streamlearn_data"  # Remplacez par votre repo HF
    repo_id = "Robin-la-Lune/streamTest"  # Remplacez par votre repo HF
    """
    
    if repo_id == "schmilblick-ai/streamlearn_data":
        # Récupération sécurisée du token dans les secrets
        try:
            hf_token = st.secrets["hf"]["HF_TOK"]
        except KeyError:
            st.error("Le token Hugging Face est introuvable dans les secrets.")
            return None
    else:
        #avoir plus tard gestion plus précise des tokens pour différents repos si besoin        
        hf_token="" #sera récupéré de manière sécurisée dans les secrets.toml, pas en dur

    

    # snapshot_download télécharge tout le dossier spécifié
    # local_dir : où le mettre sur le serveur Streamlit
    # allow_patterns : on ne prend que ce qui nous intéresse
    local_path = snapshot_download( 
        repo_id=repo_id,
        local_dir="data",                        # Racine locale
        allow_patterns=[f"{subproject_data}/*"], # Récupère tout le sous-dossier projet
        local_dir_use_symlinks=False,
        token=hf_token # ajout du token pour l'authentification
    )
    base = os.path.join("data", subproject_data) 
    st.write(f"Base path for {subproject_data}: {base} vs {local_path}") 

    if False: #pas de besoin
        # On construit le dictionnaire de chemins pour la phase suivante
        # (Adapté à votre arborescence réelle sur HF)
        base = os.path.join("data", subproject_data)
        paths = {
            "xml": os.path.join(base, "openvino_model.xml"),
            "bin": os.path.join(base, "openvino_model.bin"),
            "embeddings": os.path.join(base, "embeddings.npy")
        }
        st.success(paths)

    return local_path


# Step Decorateur pour Stockage en mémoire des traces
global _RUNTIME_TRACES
_RUNTIME_TRACES = []

def Step(description):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Préparation des mesures
            gc.collect()  # Nettoyage pour des mesures mémoires plus justes
            process = psutil.Process(os.getpid())
            
            mem_start = process.memory_info().rss / (1024**2)
            time_start = time.time()
            dt_start = datetime.now().strftime("%H:%M:%S")
            
            # Exécution de la fonction
            result = func(*args, **kwargs)
            
            # Mesures de fin
            time_end = time.time()
            mem_end = process.memory_info().rss / (1024**2)
            
            trace = {
                "Step": description,
                "Function": func.__name__,
                "Start": dt_start,
                "Duration (s)": round(time_end - time_start, 2),
                "Mem Start (MB)": round(mem_start, 2),
                "Mem End (MB)": round(mem_end, 2),
                "Mem Delta (MB)": round(mem_end - mem_start, 2),
            }
            
            # Enregistrement
            _RUNTIME_TRACES.append(trace)
            save_trace_to_disk(trace) # Enregistrement progressif
            
            return result
        return wrapper
    return decorator


#définition d'un test de validité entre deux fichiers
def is_cache_valid(source_path, cache_path):
    """Vérifie si le cache est présent et plus récent que la source."""
    if not os.path.exists(cache_path):
        return False
    # Comparaison des dates de modification
    source_mtime = os.path.getmtime(source_path)
    cache_mtime = os.path.getmtime(cache_path)
    return cache_mtime > source_mtime

# Step Decorateur pour Stockage en disque et gestion de reprise
# le temps entre la source et la target marquant doit décider
# Exemple de nom de fichier dynamique
# AVOIR filename = f"cache_{func.__name__}_u{valeur_u}_v{valeur_v}.pkl"

def Cache_Disk():
    """A voir avec dill un autre jour
    import dill

    # --- SAUVEGARDE ---
    with open('mon_modele.pkl', 'wb') as f:
        dill.dump(topic_model, f)

    # --- CHARGEMENT ---
    with open('mon_modele.pkl', 'rb') as f:
        topic_model = dill.load(f)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # 1. On récupère la signature de g
            sig = inspect.signature(func)
            
            # 2. On lie les arguments reçus (args/kwargs) à la signature
            # Cela remplit automatiquement les valeurs par défaut manquantes !
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults() 
            
            # 3. On accède aux valeurs comme dans un dictionnaire
            src_file = bound_args.arguments.get('src_file')
            tgt_file = bound_args.arguments.get('tgt_file')
            
            if is_cache_valid(src_file, tgt_file):
                # On charge les 40k x 3 coordonnées directement
                print("Chargement des data depuis le disque...")
                return joblib.load(tgt_file)
            else:
                # Exécution de la fonction
                print("☕ Sauvegarde (Préparez un café, c'est long)...")
                result = func(*args, **kwargs)
                joblib.dump(result, tgt_file)
                return result
        return wrapper
    return decorator

def save_trace_to_disk(trace, filename="./log/runtime_log.csv"):
    """Enregistre la trace ligne par ligne sur le disque."""
    df = pd.DataFrame([trace])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def show_performance_dashboard():
    st.subheader("📊 Traces d'exécution")
    runtime_log="./log/runtime_log.csv"
    if os.path.exists(runtime_log):
        df_history = pd.read_csv(runtime_log)
        
        # 1. Calcul de la fin du calcul pur
        df_history['End'] = pd.to_datetime(df_history['Start']) + pd.to_timedelta(df_history['Duration (s)'], unit='s')

        # 2. On récupère l'heure de début de l'étape SUIVANTE (le décalage)
        df_history['Next_Start'] = df_history['Start'].shift(-1)

        # 3. Le Lag (Temps de rétention / attente utilisateur)
        # C'est la différence entre le début de la suivante et la fin du calcul actuel
        # Conversion explicite en datetime

        # On remplace le dernier NaN (pour la dernière ligne) par Start+now
        df_history['Next_Start'] = df_history['Next_Start'].fillna(datetime.now().strftime('%H:%M:%S')) #pd.Timestamp.now())
    
        print(df_history, type(pd.to_datetime(df_history['Next_Start'])[0]))

        df_history['Lag_Duration'] = (pd.to_datetime(df_history['Next_Start']) \

                                      - pd.to_datetime(df_history['End'])) \
                                      .dt.total_seconds().clip(upper=3600)

        

        # On prépare un DataFrame "long" pour le graphique
        # On sépare chaque étape en deux segments : Travail et Lag
        df_melted = pd.melt(
            df_history, 
            id_vars=['Step', 'Start'], 
            value_vars=['Duration (s)', 'Lag_Duration'],
            var_name='Type_Duree', 
            value_name='Secondes'
        )

        # On renomme pour la légende
        df_melted['Type_Duree'] = df_melted['Type_Duree'].replace({
            'Duration (s)': 'Calcul (Actif)',
            'Lag_Duration': 'Attente (Lag)'
        })

        print(df_melted)

        if False:
            # Création du graphique
            fig = px.bar(
                df_melted, 
                y="Step", 
                x="Secondes", 
                color="Type_Duree",
                orientation='h',
                title="Répartition du Temps : Calcul vs Rétention Mémoire",
                color_discrete_map={
                    'Calcul (Actif)': '#FF8C00', # Orange
                    'Attente (Lag)': '#CBD5E0'   # Gris clair
                },
                # On ajoute des infos au survol
                hover_data={'Start': True, 'Secondes': ':.2f'}
            )


        if False:
            df_history['Start'] = pd.to_datetime(df_history['Start'])
            fig = px.bar(
                df_history, 
                x="Duration (s)", 
                y="Step", 
                text="Mem End (MB)", # On affiche le chiffre directement sur la barre
                orientation='h',
                title="Chronologie et Occupation RAM"
            )

            # On personnalise l'affichage du texte (ex: "850 MB")
            fig.update_traces(texttemplate='%{text} MB', textposition='outside')

            # Amélioration du design
            fig.update_layout(
                barmode='stack', 
                xaxis_title="Temps total d'occupation en mémoire (s)",
                yaxis_title=None,
                legend_title=None
            )
        
        if True:
            df_history['Start'] = pd.to_datetime(df_history['Start'])
            fig = px.area(
                df_history,
                x="Start",
                y="Mem End (MB)",
                line_shape="hv", # IMPORTANT : crée l'effet d'escalier entre les étapes
                color_discrete_sequence=['#4169E1'], # Bleu royal
                title="Évolution de la charge RAM au fil de l'eau"
            )

            # On ajoute des points pour chaque étape
            fig.add_scatter(
                x=df_history['Start'], 
                y=df_history['Mem End (MB)'], 
                mode='markers+text',
                text=df_history['Step'],
                textposition="top center",
                name="Points d'étape"
            )

        if False:
            df_history['Start'] = pd.to_datetime(df_history['Start'])
            fig = px.scatter(
                df_history,
                x="Start",
                y="Mem End (MB)",
                size="Duration (s)",      # Plus c'est gros, plus c'est long
                color="Mem Delta (MB)",    # Dégradé selon l'augmentation RAM
                text="Step",               # Nom de l'étape au-dessus du point
                title="Projection Temporelle : Charge RAM et Durée de Traitement",
                labels={"Mem End (MB)": "Mémoire Totale (MB)", "Start": "Heure d'exécution"},
                color_continuous_scale=px.colors.sequential.Viridis,
                size_max=40                # On limite la taille des bulles
            )

            # Amélioration de la visibilité du texte
            fig.update_traces(textposition='top center')
            
            # Optionnel : Ajouter une ligne qui relie les points pour voir le "chemin"
            fig.add_scatter(x=df_history['Start'], y=df_history['Mem End (MB)'], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)

        if False:
            # Métriques résumées
            col1, col2 = st.columns(2)
            col1.metric("Durée Totale", f"{df_history['Duration (s)'].sum():.1f}s")
            col2.metric("Pic Mémoire", f"{df_history['Mem End (MB)'].max():.1f} MB")
            
            # Graphique de consommation mémoire
            #st.line_chart(df_history.set_index('Step')['Mem End (MB)'])
            st.area_chart(
                df_history, 
                x='Duration (s)', 
                y='Mem End (MB)', 
                color="Step"
            )

        # Tableau détaillé
        st.dataframe(df_history)
        
        if st.button("🧹Effacer l'historique log"):
            os.remove(runtime_log)
            st.rerun()