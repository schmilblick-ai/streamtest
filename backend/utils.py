# backend/utils.py
import streamlit as st
import pandas as pd
#from io import StringIO
import re
import textwrap
from huggingface_hub import snapshot_download
import os

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