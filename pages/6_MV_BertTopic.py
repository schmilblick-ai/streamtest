import streamlit as st
import pandas as pd
#import joblib
from bertopic import BERTopic
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# Configuration de la page
st.set_page_config(page_title="Catégorisation des Avis TrustPilot", layout="wide")

# --- CHARGEMENT DES DONNÉES (Cache pour la performance) ---
@st.cache_resource
def load_models():
    """
    Charge les deux versions du modèle BERTopic.
    Assure-toi que ces dossiers existent dans ton répertoire /DS/
    """
    model_initial = BERTopic.load("dataMV/modele_bertopic_oscaro")
    model_reduit = BERTopic.load("dataMV/modele_bertopic_9themes")
    return model_initial, model_reduit


@st.cache_data
def load_data():
    # Charge ton CSV final labellisé
    df = pd.read_csv('dataMV/avis_oscaro_9_familles.csv', sep=';')
    return df

# --- APPEL DES FONCTIONS ---
# On récupère les deux modèles d'un coup
topic_model_151, topic_model_9 = load_models()
df = load_data()

# --- TITRE PRINCIPAL ---
st.title("📊 Dashboard Analyse des Avis Oscaro")
st.markdown("---")

# --- CRÉATION DES ONGLETS ---
tab1, tab2, tab3 = st.tabs(["🔍 Exploration", "📈 Data Viz", "🤖 Modélisation"])

# ---------------------------------------------------------
# TAB 1 : EXPLORATION (Nettoyage & Statistiques Brutes)
# ---------------------------------------------------------
with tab1:
    st.header("🔍 Exploration & Préparation des Données")
    st.markdown("---")

    # 1. Chargement du fichier original (pense à bien avoir 'avis_40k.csv' dans ton dossier)
    @st.cache_data
    def load_raw_data():
        return pd.read_csv('dataMV/avis_40k.csv', sep=';') # Ajuste le sep si besoin

    df_raw = load_raw_data()

    # 2. Présentation du DataFrame Original
    st.subheader("1. Données Brutes (Scrapping Trustpilot)")
    st.write(f"Le dataset contient initialement **{df_raw.shape[0]}** avis et **{df_raw.shape[1]}** colonnes.")
    st.dataframe(df_raw.head(10), use_container_width=True)

    # 3. Statistiques et Traitement
    col_stats, col_clean = st.columns(2)

    with col_stats:
        with st.expander("📊 Statistiques Descriptives", expanded=True):
            st.write(df_raw.describe(include='all'))
            st.info(f"Dimensions du tableau : {df_raw.shape}")

    with col_clean:
        with st.expander("🛠️ Nettoyage & Valeurs Manquantes", expanded=True):
            # Calcul des valeurs manquantes
            null_data = df_raw.isnull().sum()
            st.write("Valeurs manquantes par colonne :")
            st.write(null_data)
            
            st.success("""
            **Traitement effectué :**
            - Suppression des doublons (`drop_duplicates`).
            - Suppression des lignes avec commentaires vides (`isna`).
            - Remplacement des commentaires vides par leur titre non vide (`fillna`).
            """)



# ---------------------------------------------------------
# TAB 2 : DATA VIZ (Analyse des Thématiques)
# ---------------------------------------------------------
with tab2:
    st.header("📈 Analyse Sémantique Intéractive")
    st.markdown("---")

     # 1. Distribution des notes (comme demandé)
    st.subheader("1. Répartition des Notes")
    fig_notes = px.histogram(
        df, 
        x="note", 
        color="note",
        color_discrete_sequence=px.colors.diverging.RdYlGn,
        category_orders={"note": [1, 2, 3, 4, 5]}
    )
    fig_notes.update_layout(showlegend=False, xaxis_title="Note (1 à 5)", yaxis_title="Nombre d'avis")
    st.plotly_chart(fig_notes, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Comparaison des Nuages de Mots")
    
    # 2. Le Slider pour filtrer par note_max (Logique de ton Notebook)
    note_max = st.slider("Afficher les mots pour les avis ayant une note <= :", 1, 5, 5)

    # Filtrage des données
    # subset_raw utilise 'commentaire' (Notebook 1)
    # subset_clean utilise 'avis_clean' (déjà traité par ta fonction clean_text dans le Notebook 2)
    subset = df[df['note'] <= note_max]

    if not subset.empty:
        col1, col2 = st.columns(2)

        # --- NUAGE 1 : STOPWORDS STANDARDS (Notebook 1) ---
        with col1:
            st.write(f"**Texte Brut (Note <= {note_max})**")
            
            # Préparation du texte brut
            text_raw = " ".join(str(review) for review in subset.commentaire if isinstance(review, str))
            
            mots_inutiles = set(STOPWORDS)
            mots_inutiles.update(["le", "la", "les", "de", "des", "un", "une", "du", "pour", "dans", 
                                  "en", "ce", "ces", "est", "a", "au", "aux", "sur", "plus", "très"])

            wc_raw = WordCloud(
                width=800, height=500, background_color='white',
                stopwords=mots_inutiles, max_words=50,
                colormap='Reds' if note_max <= 2 else 'viridis'
            ).generate(text_raw)

            fig1, ax1 = plt.subplots()
            ax1.imshow(wc_raw, interpolation='bilinear')
            ax1.axis("off")
            st.pyplot(fig1)

        # --- NUAGE 2 : NLTK CLEAN (Ta fonction Notebook 2) ---
        with col2:
            st.write(f"**Texte Nettoyé NLTK (Note <= {note_max})**")
            
            # On utilise ta colonne 'avis_clean' (déjà traitée par clean_text)
            text_clean = " ".join(str(review) for review in subset.avis_clean if str(review) != 'nan')

            # Ta logique de couleur : plasma si note > 3, sinon Reds
            color_nltk = 'plasma' if note_max > 3 else 'Reds'

            wc_nltk = WordCloud(
                width=800, height=500, background_color='white',
                max_words=100, colormap=color_nltk
            ).generate(text_clean)

            fig2, ax2 = plt.subplots()
            ax2.imshow(wc_nltk, interpolation='bilinear')
            ax2.axis("off")
            st.pyplot(fig2)
            
        st.info(f"💡 Interprétation : Le nuage NLTK (à droite) élimine les bruits comme 'avis' ou 'plus' pour se concentrer sur les termes métier.")

    else:
        st.warning(f"Aucun avis trouvé pour la note <= {note_max}")
   
# ---------------------------------------------------------
# TAB 3 : MODELISATION (Intelligence Artificielle)
# ---------------------------------------------------------
with tab3:
    st.header("🤖 Modélisation avec BERTopic")
    st.markdown("---")

    # 1. APERÇU DU TRAITEMENT SPACY
    st.subheader("1. Nettoyage Linguistique (spaCy)")
    st.info("La lemmatisation transforme 'livraisons' en 'livrer' et 'pièces' en 'pièce' pour regrouper le sens.")
    
    # On affiche un petit comparatif avant/après
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Texte Original")
        st.write(df_raw['commentaire'].iloc[0])
    with col_b:
        st.caption("Texte Lemmatisé (spaCy)")
        st.write(df['avis_clean'].iloc[0])

    st.markdown("---")

    # 2. LES 150 THÈMES INITIAUX & HIÉRARCHIE
    st.subheader("2. Analyse de la structure initiale (151 Topics)")
    st.write("L'IA identifie initialement une multitude de micro-sujets très précis.")
    
    col1, col2 = st.columns(2)
    with col1:
        # On utilise le modèle NON réduit ici
        fig_150 = topic_model_151.visualize_topics() 
        st.plotly_chart(fig_150, use_container_width=True)
    with col2:
        # Le dendrogramme complet montre la complexité réelle
        fig_hier_150 = topic_model_151.visualize_hierarchy()
        st.plotly_chart(fig_hier_150, use_container_width=True)

    st.markdown("---")

    # 3. RÉDUCTION ET RÉSULTATS FINAUX
    st.subheader("3. Optimisation : Réduction à 9 Thèmes Stratégiques")
    st.write("Fusion des thèmes proches pour une lecture stratégique.")

    col3, col4 = st.columns(2)
    with col3:
        # On utilise le modèle RÉDUIT ici
        fig_9 = topic_model_9.visualize_topics()
        st.plotly_chart(fig_9, use_container_width=True)
    with col4:
        # Barchart des 9 thèmes finaux
        fig_bar_9 = topic_model_9.visualize_barchart(top_n_topics=9)
        st.plotly_chart(fig_bar_9, use_container_width=True)

    st.markdown("---")
    st.subheader("4. Impact Business : Satisfaction par Thématique")

    # 4. Définition du mapping (ton dictionnaire)
    map_themes = {
         -1: "Flux Standard",
         0: "Vitesse & Conformité",
         1: "Logistique & Véhicule",
         2: "Satisfaction Client",
         3: "Retards de Livraison",
         4: "Fidélité Produit/Photo",
         5: "Efficacité 'Nickel'",
         6: "Benchmark Concurrence",
         7: "Montage & Installation"
    }

    # 5. Préparation des données pour le graphique
    # On s'assure que le dataframe a bien les colonnes nécessaires
    # Note : On utilise topic_model_9 car c'est celui qui correspond à tes 9 thèmes
    df['topic_id'] = topic_model_9.topics_
    df['topic_label'] = df['topic_id'].map(map_themes).fillna("Autres/Inclassables")

    # Calcul des stats (moyenne et nombre d'avis)
    analyse_stats = df.groupby('topic_label')['note'].agg(['mean', 'count']).reset_index()
    analyse_stats = analyse_stats.sort_values(by='mean')

    # 6. Affichage des deux éléments côte à côte
    col_table, col_chart = st.columns([1, 2])

    with col_table:
        st.write("**Tableau récapitulatif**")
        # On arrondit la moyenne pour la lecture
        display_stats = analyse_stats.copy()
        display_stats['mean'] = display_stats['mean'].round(2)
        st.dataframe(display_stats, use_container_width=True)
        st.caption(f"Total des avis classés : {len(df)}")

    with col_chart:
        import seaborn as sns
        
        # Configuration du graphique Seaborn
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Création du barplot avec ta palette divergente
        ax = sns.barplot(
            x='mean', 
            y='topic_label', 
            data=analyse_stats, 
            palette='RdYlGn'
        )

        # Ajout de la ligne d'objectif Qualité
        plt.axvline(x=4, color='blue', linestyle='--', label='Objectif Qualité (4/5)')
        
        plt.title('Satisfaction Client par Thématique', fontsize=14)
        plt.xlabel('Note Moyenne (1 à 5)')
        plt.ylabel('') # On vide le label Y car les noms sont explicites
        plt.xlim(1, 5)
        plt.legend(loc='lower right')

        # Affichage dans Streamlit
        st.pyplot(plt)

    st.success("🎯 **Analyse :** Les thématiques en rouge/orange (sous la ligne bleue) sont celles où Oscaro doit concentrer ses efforts d'amélioration.")
