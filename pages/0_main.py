import streamlit as st
from SL_app01 import main_header
from backend.utils import sync_project_files

main_header()

st.markdown("""
Navigue dans le menu à gauche :

- **Mots proches** - wordcloud des voisins sémantiques
- **Analogies** - A − B + C = ?
- **Clustering** - visualisation TSNE des embeddings
- **Outliers** - quel mot n'appartient pas au groupe ?
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
    ```bash        
    sur [github](https://github.com/schmilblick-ai/)  
    Ne rien cocher (no README, no .gitignore) 
    
    # Ajouter tous les fichiers
    git add .

    # Premier commit
    git commit -m "init : application Word2Vec Streamlit"    

    # Renommer la branche principale
    git branch -M main

    # Prepare credentials        
    ssh-keygen -t ed25519 -C "ton@email.com"
    #do not overide, if the key exists, take that key
    # Copier la clé publique et ma coller dans github si pas fait (→ Settings → SSH and GPG keys → New SSH key)
    cat ~/.ssh/id_ed25519.pub        
            
    # Relier le repo local au repo GitHub
    # NON PAS https - préféré ssh # git remote add origin https://github.com/TON_USERNAME/streamtest.git
    # En changeant l'URL pour SSH and check
    git remote set-url origin git@github.com:TON_USERNAME/streamtest.git
    git remote -v        

    #  V E R I F I C A T I O N
    git remote -v
    # origin  https://github.com/TON_USERNAME/streamtest.git (fetch)
    # origin  https://github.com/TON_USERNAME/streamtest.git (push)

    git status
    # On branch main - nothing to commit        

                                
    # Premier push
    git push -u origin main        

    ```    
    
    (le modèle entraîné sauvegardé au format H5 et le fichier `.py` avec le script Streamlit).

    > **Note :** Si les poids d'un modèle de Deep Learning sont trop volumineux, le fichier H5 ne peut pas être déposé sur le repo GitHub
    >
    > Dans ce cas, nous pouvons utiliser **Git Large File Storage (Git LFS)**.
    >

    Pour utiliser Git LFS - commenter les .gitignore pour les fichiers data et préparer le .gitattributes:
    ```bash
    git lfs install
    git clone
    git lfs track "*.keras"
    git lfs track "*.h5"
    git lfs track "*.pkl"
    git lfs track "*.csv"

    git add .gitattributes
    git add fichier.h5

    #Vérifier que LFS est bien en charge
    git lfs ls-files
    # doit lister tes fichiers lourds

    git commit -m "fichier lourds"
    git push
    ```

    ## Déploiement via Colab ngrok - might not be suitable for uv

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
    
    ## Déploiement streamlit cloud via uv + github
    ```bash     
    # 1. préparer le repo
    # 2. vérif dépendances pyproject.toml avec la section tool.uv         
    # 3. Générer le lockfile 
    uv sync
            
    # 4. streamlit cloud et uv ? pip est par defaut sur streamlit cloud, pour uv il lui faut une config
    # see .streamlit/config.toml et packages.txt
    
    # 5. et la génération du requirements.txt par uv
    uv pip compile pyproject.toml -o requirements.txt        
    
    # 6. pousser sur github
    git add pyproject.toml uv.lock requirements.txt .python-version
    git commit -m "config : dépendances uv + requirements Streamlit"
    git push            

    sur share.streamlit.io 
    7. Déployer sur Streamlit Cloud
    Va sur share.streamlit.io :

        Connect → connecte ton compte GitHub
        New app
        Renseigne :

        Repository : TON_USERNAME/streamtest
        Branch : main
        Main file : main.py
        # Deploy → Streamlit installe les dépendances et lance l'app
            
        Your app is in the oven - https://streamtest-wkre8ddner938gx4ezeils.streamlit.app/
        
        It rocks all 4 pages
    

    ``` 



    ## D A Y  -  T O  -  D A Y  -  O P E R A T I O N
    ```bash
    #adding a new library -> update dependencies
    uv add newlib
    uv pip compile pyproject.toml -o requirements.txt

    # Pousser
    git add .
    git commit -m "feat : nouvelle fonctionnalité"
    git push
    # Streamlit Cloud redéploie automatiquement !         

    #Pour les modèles lourds
    # Si model.keras est dans le repo
    git lfs track "*.keras"
    git lfs track "*.pkl"
    git add .gitattributes
    ? git add data/.*
    git commit -m "config : Git LFS pour les modèles"
    git push        

    ```     
    💡 Le uv.lock est important à committer même si Streamlit Cloud utilise requirements.txt — il garantit la reproductibilité sur ta machine locale. 
            
    Le workflow idéal : uv pour développer localement, requirements.txt généré par uv pour le déploiement Streamlit Cloud.        

            

    Maintenant vous maîtrisez Streamlit !


    Hep, et le debugging ? Aaah, c'est ici https://ploomber.io/blog/streamlit-debugging/  ... hélas ca ne marche pas 😡😭😤 !

    """)

if True:
    import os

    # Création automatique des dossiers au lancement ici juste data qui n'est plus dans github
    os.makedirs("data", exist_ok=True)

    
    
    # R E C U P  H F  Presque comme des libnames avec une synchro
    # Voilà avec une boucle c'est sympa !
    for Proj in ["dataMV","dataLG"]:
        if not os.path.exists(f"data/{Proj}"):

            st.success("Données déjà présentes localement, pas besoin de les recharger depuis Hugging Face !")
            with st.spinner(f"Récupération des ressources pour {Proj} depuis Hugging Face... [{Proj}_Path]"):
                #"Variable globale {Proj}_Path créée avec le chemin local synchronisé"
                globals()[f"{Proj}_Path"] = sync_project_files(Proj) 
                st.success(f"ressources pour {Proj} chargés !")
        else:
            st.success(f"Données pour {Proj} déjà présentes localement, pas besoin de les recharger depuis Hugging Face !")
            globals()[f"{Proj}_Path"] = f"data/{Proj}"    

    # OPTIM on pourra mettre chaque load dans la page qui en a besoin, mais pour l'instant c'est plus simple de tout faire d'un coup
    # pour voir, et ça évite les problèmes de chargement à la volée dans les pages



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
        
        !streamlit run /content/streamlit_app.py & npx localtunnel - port 8501

        
        Maintenant vous maitrisez Streamlit et ses bonnes pratiques pour tout type de projet et pour tout environnement de travail !

        </pre>
        """, unsafe_allow_html=True)
    
