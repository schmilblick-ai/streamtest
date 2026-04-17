import streamlit as st
import pandas as pd
#import joblib
from bertopic import BERTopic
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
#import plotly.express as px
import matplotlib.pyplot as plt
#import spacy
import fr_core_news_sm
# N O T   I N   U S E

# Configuration de la page
st.set_page_config(page_title="Catégorisation des Avis TrustPilot", layout="wide")

# --- CHARGEMENT DES DONNÉES (Cache pour la performance) ---
@st.cache_resource
def load_models():
    """
    Charge les deux versions du modèle BERTopic.
    Assure-toi que ces dossiers existent dans ton répertoire /DS/
    """
    model_initial = BERTopic.load("modele_bertopic_oscaro")
    model_reduit = BERTopic.load("modele_bertopic_9themes")
    return model_initial, model_reduit


@st.cache_data
def load_data():
    # Charge ton CSV final labellisé
    df = pd.read_csv('avis_oscaro_9_familles.csv', sep=';')
    return df

@st.cache_resource
def load_nlp():
    # Charge le modèle via le package directement
    return fr_core_news_sm.load()

nlp = load_nlp()

def lemmatize_single_text(text):
    """Version pour un seul texte (utilisée dans le module de test)"""
    if not text or text.strip() == "":
        return ""
    doc = nlp(text.lower())
    # La logique : pas de stop words, pas de ponct, longueur > 2
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 2]
    return " ".join(tokens)

# --- APPEL DES FONCTIONS ---
# On récupère les deux modèles d'un coup
topic_model_151, topic_model_9 = load_models()
df = load_data()

# --- TITRE PRINCIPAL ---
st.title("📊 Dashboard Analyse des Avis Oscaro")
st.markdown("---")

# --- CRÉATION DES ONGLETS ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Exploration", "📈 Data Viz", "🤖 Modélisation", "🔮 Prédiction"])

# ---------------------------------------------------------
# TAB 1 : EXPLORATION (Nettoyage & Statistiques Brutes)
# ---------------------------------------------------------
with tab1:
    st.header("🔍 Exploration & Préparation des Données")
    st.markdown("---")

    # 1. Chargement du fichier original (pense à bien avoir 'avis_40k.csv' dans ton dossier)
    @st.cache_data
    def load_raw_data():
        return pd.read_csv('avis_40k.csv', sep=';') # Ajuste le sep si besoin

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
            
        st.info("💡 Interprétation : Le nuage NLTK (à droite) élimine les bruits comme 'avis' ou 'plus' pour se concentrer sur les termes métier.")

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
         7: "Montage & Installation",
         8: "SAV ou Autres"
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


   
# ---------------------------------------------------------
# TAB 4 : DEMO DATA (Exploration des 40 000 avis)
# ---------------------------------------------------------
with tab4:
    st.header("📋 Exploration des avis classés")
    st.markdown("---")

    # 1. Préparation des colonnes à supprimer
    cols_to_drop = ['url', 'textServiceReply', 'dateServiceReply', 'DataServiceReply']
    
    # On crée une copie pour ne pas impacter les autres onglets
    # On ne garde que les colonnes existantes pour éviter les erreurs
    df_display = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. Barre latérale de filtres (Filtres interactifs)
    st.subheader("🛠️ Filtres de recherche")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        # Filtre par thématique (Label)
        liste_labels = ["Tous"] + list(df_display['topic_label'].unique())
        choix_label = st.selectbox("Filtrer par Thématique :", liste_labels)
    
    with col_f2:
        # Filtre par note
        choix_note = st.multiselect("Filtrer par Note :", 
                                    options=[1, 2, 3, 4, 5], 
                                    default=[1, 2, 3, 4, 5])
    
    with col_f3:
        # Recherche par mot clé
        search_query = st.text_input("Rechercher un mot spécifique :", "")

    # 3. Application des filtres
    df_filtered = df_display.copy()

    if choix_label != "Tous":
        df_filtered = df_filtered[df_filtered['topic_label'] == choix_label]
    
    if choix_note:
        df_filtered = df_filtered[df_filtered['note'].isin(choix_note)]
    
    if search_query:
        # Recherche insensible à la casse dans la colonne commentaire
        df_filtered = df_filtered[df_filtered['commentaire'].str.contains(search_query, case=False, na=False)]

    # 4. Affichage du tableau
    st.markdown(f"**Nombre d'avis correspondants :** {len(df_filtered)}")
    
    # Utilisation de dataframe avec configuration pour une meilleure lecture
    st.dataframe(
        df_filtered,
        use_container_width=True,
        column_config={
            "note": st.column_config.NumberColumn("Note", format="%d ⭐"),
            "commentaire": st.column_config.TextColumn("Commentaire", width="large"),
            "topic_label": st.column_config.TextColumn("Catégorie IA"),
            "avis_clean": None # On peut aussi cacher la version nettoyée si on veut
        },
        hide_index=True
    )

    # 5. Option de téléchargement
    csv = df_filtered.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button(
        label="📥 Télécharger la sélection en CSV",
        data=csv,
        file_name='avis_filtres.csv',
        mime='text/csv',
    )
    # ---------------------------------------------------------
    # 5. PRÉDICTION EN TEMPS RÉEL
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 Tester l'IA : Classification d'un nouvel avis")
    
    with st.expander("Cliquez ici pour tester un commentaire", expanded=True):
        user_input = st.text_area(
            "Saisissez un avis client (ex: 'Ma commande est arrivée en retard et le carton était abîmé')",
            placeholder="Tapez ici..."
        )

        if user_input:
            with st.spinner('Analyse linguistique...'):
                text_clean = lemmatize_single_text(user_input)
            
            if not text_clean:
                st.warning("⚠️ Le texte est trop court ou ne contient que des mots vides après nettoyage.")
            else:
                st.write(f"**L'IA analyse ces mots-clés :** `{text_clean}`")
            
                # 1. Prédiction
                topics, probs = topic_model_9.transform([text_clean])
                
                # --- LOGIQUE DE DÉCISION ---
                if hasattr(probs[0], "__len__"):
                    import numpy as np
                    # On cherche l'ID qui a la plus haute probabilité dans le vecteur
                    best_id = np.argmax(probs[0]) 
                    confidence = float(probs[0][best_id])
                    
                    # Si le modèle renvoie -1 (bruit) mais qu'on a un score correct ailleurs (> 0.25)
                    # On "force" la thématique la plus probable
                    if topics[0] == -1 and confidence > 0.25:
                        predicted_id = best_id
                    else:
                        predicted_id = topics[0]
                else:
                    predicted_id = topics[0]
                    confidence = float(probs[0])
                # ----------------------------------
                
                # 2. ON DÉFINIT LE LABEL (C'est cette ligne qu'il te manque ou qui est placée trop bas)
                label = map_themes.get(predicted_id, "Inclassable / Bruit")

                # 3. Gestion de la confiance (Sécurité si multiprobs activé)
                if hasattr(probs[0], "__len__"):
                    # Si c'est un tableau, on prend la proba du topic prédit (si >= 0)
                    confidence = float(probs[0][predicted_id]) if predicted_id != -1 else 0.0
                else:
                    # Si c'est un seul float
                    confidence = float(probs[0])
        
                # --- ON RÉPARER LE GRAPHIQUE ---
                if hasattr(probs[0], "__len__"):
                    # On crée les labels dynamiquement
                    labels_graphique = [map_themes.get(i, f"Thème {i}") for i in range(len(probs[0]))]
                    
                    # On crée le DataFrame pour Plotly
                    df_probs = pd.DataFrame({
                        'Thématique': labels_graphique,
                        'Score': probs[0]
                    }).sort_values(by='Score', ascending=True)
                # ---------------------------------------------------------
                
                # 4. Affichage des résultats principaux
                col_res1, col_res2 = st.columns([1, 3])
                
                with col_res1:
                    st.metric("Thématique détectée", label)
                
                with col_res2:
                    if predicted_id == -1 or confidence < 0.35:
                        st.warning(f"Confiance faible ({round(confidence*100, 1)}%). Classé en Flux Standard.")
                    else:
                        st.success(f"L'avis a été classé dans la catégorie : **{label}**")
                        st.progress(confidence)
                        st.caption(f"Indice de confiance : {round(confidence * 100, 2)}%")
                    
                    # --- GRAPHIQUE DES PROBABILITÉS ---
                    if hasattr(probs[0], "__len__"):
                        st.write("📈 **Répartition des probabilités par thématique**")
                        
                        # UTILISER .get() ICI AUSSI POUR ÉVITER LE CRASH
                        labels_pour_graphique = [map_themes.get(i, f"Thème {i}") for i in range(len(probs[0]))]
                        
                        df_probs = pd.DataFrame({
                            'Thématique': labels_pour_graphique,
                            'Score': probs[0]
                        }).sort_values(by='Score', ascending=True)

                        fig_all_probs = px.bar(
                            df_probs, 
                            x='Score', 
                            y='Thématique', 
                            orientation='h',
                            color='Score',
                            color_continuous_scale='Blues',
                            text_auto='.2f'
                        )

                        fig_all_probs.update_layout(
                            height=350, 
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title="Niveau de correspondance",
                            yaxis_title=""
                        )
                        
                        st.plotly_chart(fig_all_probs, use_container_width=True)