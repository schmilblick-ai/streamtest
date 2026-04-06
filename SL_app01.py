import streamlit as st
from backend.utils import load_css

       
st.set_page_config(
    page_title="Word2Vec Explorer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

st.title("Word2Vec Explorer — Avis cinéma")
st.caption("Exploration des embeddings entraînés sur une base d'avis de films")



st.markdown("""
Navigue dans le menu à gauche :

- **Mots proches** — wordcloud des voisins sémantiques
- **Analogies** — A − B + C = ?
- **Clustering** — visualisation TSNE des embeddings
- **Outliers** — quel mot n'appartient pas au groupe ?
""")

st.markdown("""
    ## Déploiement de l'application Streamlit via GitHub

    ### Nous voulons héberger notre application Streamlit. Pour cela, nous avons plusieurs possibilités :

    [Structure Streamlit](https://towardsdatascience.com/how-to-structure-and-organise-a-streamlit-app-e66b65ece369/)

    - via GitHub
    - via Colab
    
    Nous essayons avec GitHub.

    (o) Suivre les étapes du notebook précédent pour héberger l'application Streamlit à partir de GitHub.
    
    Créer notamment un repository GitHub et y déposer les fichiers nécessaires à l'application Streamlit
    
    (le modèle entraîné sauvegardé au format H5 et le fichier `.py` avec le script Streamlit).

    > **Note :** Si les poids d'un modèle de Deep Learning sont trop volumineux, le fichier H5 ne peut pas être déposé sur le repo GitHub
    >
    > Dans ce cas, nous pouvons utiliser **Git Large File Storage (Git LFS)**.
    >

    Pour utiliser Git LFS :
    ```bash
    git lfs install
    git clone
    git lfs track "*.h5"
    git add .gitattributes
    git add fichier.h5
    git commit -m fichier.h5
    git push
    ```

    ## Déploiement via Colab

    Installer Streamlit et ngrok :
    ```python
    !pip install -q streamlit
    !pip install pyngrok
    ```

    Puis connecter ngrok :
    ```python
    !./ngrok authtoken token

    from pyngrok import ngrok
    public_url = ngrok.connect(port='8501')
    public_url
    ```

    Créer le fichier app :
    ```python
    %%writefile streamlit_app.py
    import streamlit as st
    ```

    Lancer l'application :
    ```bash
    !streamlit run /content/streamlit_app.py & npx localtunnel --port 8501
    ```

    Maintenant vous maîtrisez Streamlit !
    """)

if False:
    st.markdown("""
        <pre>        
        ## Déploiement de l'application Streamlit via GitHub
        
        ### Nous voulons héberger notre application Streamlit. Pour cela, nous avons plusieurs possibilitésss :               
        [structure streamlit](https://towardsdatascience.com/how-to-structure-and-organise-a-streamlit-app-e66b65ece369/)
                
        - via GitHub
        - via Colab
        - Nous essayons avec GitHub.

        o Suivre les étapes du notebook précédent pour héberger l'application Streamlit à partir de GitHub. 
        Créer notamment un repository GitHub et y déposer les fichiers nécessaires à l'application Streamlit 
        (le modèle entrainé sauvegardé au format H5 et le fichier .py avec le script Streamlit).
                
        A noter que si les poids d'un modèle de Deep Learning sont trop volumineux, le fichier H5 contenant le modèle ne peut pas être déposé sur le repo GitHub. 
        <blink>Dans ce cas, nous pouvons utiliser Git Large File Storage (Git LFS)</blink>, une fonctionnalité de Git permettant de stocker des fichiers lourds dans un dépôt distant. Cette fonctionnalité permet ainsi d'utiliser le flux de travail Git quelques soient les fichiers utilisés (données volumineuses, vidéos, songs, poids d'un modèle).

        Pour utiliser Git LFS, il faut d'abord télécharger l'extension de commandes Git.

        Puis :
        
        - git lfs install         #pour installer Git LFS
        - git clone               #pour cloner le repository GitHub
        - git lfs track "*.h5"    #pour utiliser Git LFS sur des fichiers en format H5
        - git add .gitattributes
        
        Enfin, il faut utiliser les commandes Git classiques :
                
        - git add fichier.h5
        - git commit -m fichier.h5
        - git push
        
        ## Déploiement de l'application Streamlit via Colab
        
        Dans le cas où nous travaillons sur Google Colab, plutôt que de télécharger nos fichiers et de créer un repo GitHub, nous pouvons directement 
        déployer une application Streamlit via Colab grâce à ngrok. 
        ngrok est un proxy permettant de passer d'un URL public à un réseau privé (dans notre cas notre ordinateur en local).

        Pour utiliser Streamlit via Google Colab avec ngrok, il faut suivre ces différentes étapes :

        ## Installer Streamlit et ngrok sur Google Colab avec
        
        `!pip install -q streamlit
        !pip install pyngrok`
        
        Créer un compte ngrok sur le site officiel
        Copier le token disponible dans l'onglet "Your Authtoken"
        Executer le code suivant sur Google Colab :

        
        `!./ngrok authtokens token #où token est le token copié

        from pyngrok import ngrok 
        public_url = ngrok.connect(port='8501')
        public_url`
        

        
        Executer le code suivant dans une nouvelle cellule Google Colab pour créer un fichier .py (appelé streamlit_app.py ici). Puis écrire le script Python associé à l'application Streamlit dans la cellule.
        

        `%%writefile streamlit_app.py 
        import streamlit as st 
        #Insérer code Python contenant les commandes Streamlit`
        
        Une fois que le code est terminé, executer le code suivant dans une nouvelle cellule Google Colab pour déployer l'application Streamlit.
        
        !streamlit run /content/streamlit_app.py & npx localtunnel — port 8501

        
        Maintenant vous maitrisez Streamlit et ses bonnes pratiques pour tout type de projet et pour tout environnement de travail !

        </pre>
        """, unsafe_allow_html=True)