import os
import gc
import sys
import umap
import numba
import psutil
import joblib
import sklearn
import numpy as np
import pandas as pd
import streamlit as st
from backend.utils import load_css
# Import du décorateur depuis le sous-dossier
from backend.utils import Step, show_performance_dashboard

#st.header('Basketball')

# Sidebar navigation
#
#st.sidebar.page_link('pages/1_mot_proches.py', label='Basketball')
st.set_page_config(
    page_title="Word2Vec Explorer",
    page_icon="😎", #"🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()

#print(st.page)
#st.title("Word2Vec Explorer - Avis cinéma")
#st.caption("Exploration des embeddings entraînés sur une base d'avis de films")


## central various header definition - for incubation of various proposals
def cinoch_header():
    st.title("Word2Vec Explorer - Avis cinéma")
    st.caption("Exploration des embeddings entraînés sur une base d'avis de films")   

def main_header():
    st.title("🔬Explore TrustPilot multilabel classifier")
    st.caption("Exploration streamlit embeddings Trustpilot, embeddings d'avis cinoche et structure server")   

def osexplo_header():
    st.title("os exploration | Query the bar metal and review os characteristics")
    st.caption("Run adhoc bash commands and understant underlying server's layout")   

def Marvin_header():
    st.title("Marvin outstanding proposals")
    st.caption("Review of assets provided by MLE Marvin on the BertTopic modelling")   

def Lionel_header():
    st.title("Lionel sharpening multimodal classifier")
    st.caption("Review of assets provided by MLE Lionel on the BertTopic alternatives")   

def Robin_header():
    st.title("Robin advanced alternatives")
    st.caption("Review of assets provided by MLE Robin on the BertTopic modelling")   

def Bestof_collection_header():
    st.title("Overall team proposal and consolidation")
    st.caption("Review of assets provided by MLE Marvin on the BertTopic modelling")   

#if False:
 #   st.write(f"where are we {dir(pg._page)} {pg._page}")
 #   st.write(f"where are we {pg._page.parent},{pg._page} {outliers} {dir(outliers)} {outliers._url_path}")
    ##############################################################################################

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Conversion en MB

# 1. La fonction d'observabilité décorée en tant que Dialog
@st.dialog("Inspection de la Mémoire")
def show_memory_stats():
    st.write("🔍 **Analyse des objets en cours...**")
    
    # Calcul de la RAM globale
    mem_total = psutil.Process().memory_info().rss / (1024 * 1024)
    st.metric("Consommation RAM Totale", f"{mem_total:.2f} MB")
    
    # Listing des objets lourds
    data = []
    for name, value in globals().items():
        if not name.startswith('_'):
            size = sys.getsizeof(value) / (1024 * 1024)
            if size > 0.0:  # On ne montre que ce qui fait plus de 100 KB
                data.append({"Variable": name, "Taille (MB)": size})
    
    if data:
        df = pd.DataFrame(data).sort_values("Taille (MB)", ascending=False)
        st.table(df)
    else:
        st.info("Aucun objet lourd détecté dans l'espace global.")

    # Bouton d'action à l'intérieur de la popup
    if st.button("🧹 Nettoyer la mémoire (GC)"):
        gc.collect()
        st.success("Garbage Collector lancé !")
        st.rerun()

# CSS pour fixer le container en haut
st.markdown("""
    <style>
        div[data-testid="stVerticalBlock"] > div:has(div.fixed-header) {
            position: sticky;
            top: 1px;
            background-color: #04da8d;
            z-index: 999;
            padding-bottom: 10px;
            border-bottom: 1px solid ; //#ddd;
        }
    </style>
""", unsafe_allow_html=True)

# Création du container avec une classe spécifique
header = st.container(height=120)
with header:
    # On ajoute un div vide avec la classe 'fixed-header' pour que le CSS le cible
    st.markdown('<div class="fixed-header"></div>', unsafe_allow_html=True,)

    if True:
        metrics_data = [("Cluster A", "12k","+50MB"), ("Cluster B", "8k", "-3MB")]#, ("Cluster C", "20k", "")]

        # Création dynamique de N colonnes
        nbMetrics = len(metrics_data)
        cols = st.columns(spec=[0.6 if i == 0 else 0.4/nbMetrics for i in range(1+nbMetrics)])  # La première colonne est plus large pour le titre
        
        cols[0].title("✨ Analyse des 40k Avis")
        mem_usage = get_memory_usage()
        cols[1].write("")
        cols[2].metric("RAM Utilisée", f"{mem_usage:.2f} MB")
        if False:
            for i, (label, value, delta) in enumerate(metrics_data):
                cols[1+i].metric(label, value, delta,height=64)

        #col2.metric("Mémoire vive", "1.1 GB", "-50MB", )


##############################################################################################
# Create a sidebar selection
##############################################################################################

main        = st.Page("pages/0_main.py"           , title="main", icon=":material/dashboard:")
mot_proches = st.Page("pages/1_mot_proches.py"    , title="mot proches", icon=":material/dashboard:")
analogie    = st.Page("pages/2_analogies.py"      , title="analogie", icon=":material/bug_report:")
clustering  = st.Page("pages/3_clustering.py"     , title="clustering", icon=":material/notification_important:")
outliers    = st.Page("pages/4_outliers.py"       , title="outliers", icon=":material/search:")
osexplo     = st.Page("pages/5_osexplo.py"        , title="os Exploration", icon=":material/history:",)
Marvin      = st.Page("pages/6_MV_BertTopic.py"   , title="Marvin", icon=":material/history:",)
Lionel      = st.Page("pages/7_LG_BertTopic.py"   , title="Lionel", icon=":material/history:",)
#Robin      = st.Page("pages/8_RM_multiclassif.py", title="Robin", icon=":material/history:",)
Robin       = st.Page("pages/8_RM_clustering.py"  , title="Robin", icon=":material/history:",)
Whole       = st.Page("pages/9_wholeClassif.py"   , title="BestOf", icon=":material/history:",)

with st.sidebar:
    st.logo("https://cdn.trustpilot.net/brand-assets/4.3.0/logo-white.svg",size="large")
    st.sidebar.markdown("""
        <style>
            [data-testid="stSidebarHeader"]::after {
                content: "eXploreR";
                //display: block;
                margin-bottom: 0px;        
                margin-bottom: 0px;
                font-weight: bold;
                font-size: 25px;
                color: #a78bfa;
            }
        </style>
    """, unsafe_allow_html=True)
    
    #st.page_link(main, label='🏠Home',)
      
    # Affichage dans la sidebar pour monitoring constant
    with st.sidebar.expander("✨ Admin Panel", expanded=False):
        st.title("🚀 Observabilité")
        mem_usage = get_memory_usage()
        st.metric("RAM Utilisée", f"{mem_usage:.2f} MB")
        st.sidebar.metric("RAM Utilisée", f"{mem_usage:.2f} MB")
        # Alerte visuelle si on approche de la limite (souvent 1GB sur le tier gratuit)
        if mem_usage > 800:
            st.warning("⚠️ Attention : Saturation RAM proche !")
        
        st.divider()

        # 2. Le bouton de déclenchement dans la Sidebar
        
        if st.button("📊 Inspecter les ressources"):
            show_memory_stats()
        if st.button("📈 Dashboard de performance"):
            show_performance_dashboard()

        if st.button("🧹 Garbage Collector"):
            gc.collect()
            st.cache_data.clear()  # Optionnel : on peut aussi vider le cache de Streamlit pour libérer plus de mémoire
            st.success("Garbage Collector lancé !")
            st.rerun()
        st.page_link(osexplo, label='🏆 Os Exploration', icon=":material/emoji_events:",)

        st.divider()
        st.title("⚙️ Environnement")
        # Version de Python
        st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Versions des bibliothèques critiques
        st.write("**Dépendances :**")
        st.text(f"umap-learn: {umap.__version__}")
        st.text(f"numba:      {numba.__version__}")
        st.text(f"joblib:     {joblib.__version__}")
        st.text(f"scikit-learn: {sklearn.__version__}")

pg=st.navigation({"main":[main]
    #"🎬Cinéma": [mot_proches, analogie, clustering, outliers]  
    ,"🧑‍✈️TrustP": [Marvin, Lionel, Robin, Whole]
    ,"⚙️System": [osexplo]
    } ,)
pg.run()


