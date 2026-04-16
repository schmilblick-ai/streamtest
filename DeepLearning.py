
## Création du fichier Python pour Streamlit
### Maintenant que le modèle de Deep Learning a été entrainé et sauvegardé, nous pouvons passer au Streamlit. Comme d'habitude, nous utilisons un éditeur de code Python (par exemple VSCode ou Spyder) pour obtenir un fichier .py.
"""
(h) Créer un fichier .py qui contiendra le script Python dédié à l'application Streamlit. 
L'enregistrer dans le même dossier que les autres fichiers du projet.

(i) Donner un titre au Streamlit.

(j) Charger les poids du modèle enregistré en utilisant la méthode load_weights de Keras.
"""

import streamlit as st, pandas as pd  # noqa: E401
from keras import Sequential
from keras.layers import Embedding, Dense, GlobalAveragePooling1D
from  pathlib import Path
import numpy as np  # noqa: E402
from sklearn.preprocessing import Normalizer  # noqa: E402



@st.cache_resource
def load_df():
    return pd.read_csv("./data/MovieReview_gld.csv")

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    try:
        tgtidx=word2idx[word]
        answ={}
        if tgtidx <= len(vectors):
            index_closest_words = find_closest(tgtidx, vectors, number)
            for index_word in index_closest_words :
                #answ.append(",".join([idx2word[index_word[1]]," -- ",str(index_word[0])]))
                #answ.append([idx2word[index_word[1]],str(index_word[0])])
                answ[idx2word[index_word[1]]]=str(index_word[0])
            return answ    
        else:
            #return f"{word} of index={tgtidx} is too unfrequent"
            return {f"{word}",f" of index={tgtidx} is too unfrequent"}
            
    except KeyError as e:
        return {f"{word}",f"a KeyError - reason the work {str(e)} in not in word space" }


def test_reproductibilite(X_sample, y_sample):
    model = tf.keras.models.load_model('model.keras')
    predictions = model.predict(X_sample)
    
    # Vérifie que les prédictions sont cohérentes
    assert predictions.shape == y_sample.shape
    print(f"Exemple prédit : {predictions[0]}")
    print(f"Valeur réelle  : {y_sample[0]}")



#retrieve reviews gld - this means prepared and optimized
df = load_df()

#retrieval of tokenizer and vocabsize - avoiding fit_on_texts - using the .pkl is better
import tensorflow as tf  # noqa: E402
import pickle  # noqa: E402


#retrieval of lookups
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = tokenizer.num_words

embedding_dim = 300

#retrieve and cache the model
model = load_model("./data/word2vec.keras")

### La similitude est une métrique mesurant la distance entre deux mots. Cette distance représente la façon dont les mots sont liés entre eux.

### (k) Ajouter le code suivant pour extraire la matrice d'embeddings et définir les fonctions de similitude.
vectors = model.layers[0].trainable_weights[0].numpy()

#test_reproductibilite(X_test[:5], y_test[:5])

### (l) Créer des widgets Streamlit permettant à l'utilisateur d'afficher les 10 mots les plus proches d'un mot choisi, grâce à la fonction print_closest définie précédemment.

### Remarque : Une idée pour rendre le Streamlit encore plus intéractif pourrait être de laisser à l'utilisateur le choix du nombre de mots proches.
### #Exemple d'utilisation de la fonction print_closest
### print_closest('zombie')
### (m) Créer des widgets Streamlit pour jouer sur les propriétés sémantiques et arithmétiques d'un mot, préservées par le modèle Word2Vec. La fonction compare définie précedemment pourra être utilisée.
### (n) Personnaliser le Streamlit.

# S T R E A M L I T
st.title("Modèle Word2Vec")
st.sidebar.title("Sommaire")
pages=["Display 10 words", "WordCloud ?"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("### Introduction")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())
    
    NBWORDS=    st.slider("Choose Number of words",min_value=1,max_value=20,value=10,step=1)
    
    text=st.text_input(label="mon input")
    if st.button(f"Search {NBWORDS} closest"):
        st.write(print_closest(text,NBWORDS))

    message = st.chat_input("")
    st.write(f"you requested {message}",)
    if message:
        text=st.write(print_closest(message,NBWORDS))



    st.markdown("""
        ## Déploiement de l'application Streamlit via GitHub
        
        ### Nous voulons héberger notre application Streamlit. Pour cela, nous avons plusieurs possibilitésss :
                
        <a href="https://towardsdatascience.com/how-to-structure-and-organise-a-streamlit-app-e66b65ece369/" > structure streamlit</a>
                
        - via GitHub
        - via Colab
        - Nous essayons avec GitHub.

        ### (o) Suivre les étapes du notebook précédent pour héberger l'application Streamlit à partir de GitHub. Créer notamment un repository GitHub et y déposer les fichiers nécessaires à l'application Streamlit (le modèle entrainé sauvegardé au format H5 et le fichier .py avec le script Streamlit).
        ### A noter que si les poids d'un modèle de Deep Learning sont trop volumineux, le fichier H5 contenant le modèle ne peut pas être déposé sur le repo GitHub. Dans ce cas, nous pouvons utiliser Git Large File Storage (Git LFS), une fonctionnalité de Git permettant de stocker des fichiers lourds dans un dépôt distant. Cette fonctionnalité permet ainsi d'utiliser le flux de travail Git quelques soient les fichiers utilisés (données volumineuses, vidéos, songs, poids d'un modèle).

        ### Pour utiliser Git LFS, il faut d'abord télécharger l'extension de commandes Git.

        ### Puis :

        git lfs install pour installer Git LFS
        git clone pour cloner le repository GitHub
        git lfs track "*.h5" pour utiliser Git LFS sur des fichiers en format H5
        git add .gitattributes
        Enfin, il faut utiliser les commandes Git classiques :

        git add fichier.h5
        git commit -m fichier.h5
        git push
        </pre>
        ## Déploiement de l'application Streamlit via Colab
        <pre>
        Dans le cas où nous travaillons sur Google Colab, plutôt que de télécharger nos fichiers et de créer un repo GitHub, nous pouvons directement 
        déployer une application Streamlit via Colab grâce à ngrok. 
        ngrok est un proxy permettant de passer d'un URL public à un réseau privé (dans notre cas notre ordinateur en local).

        Pour utiliser Streamlit via Google Colab avec ngrok, il faut suivre ces différentes étapes :

        ## Installer Streamlit et ngrok sur Google Colab avec
        </pre>
        `!pip install -q streamlit
        !pip install pyngrok`
        <pre>
        Créer un compte ngrok sur le site officiel
        Copier le token disponible dans l'onglet "Your Authtoken"
        Executer le code suivant sur Google Colab :

        </pre>
        `!./ngrok authtokens token #où token est le token copié

        from pyngrok import ngrok 
        public_url = ngrok.connect(port='8501')
        public_url`
        </pre>

        <pre>
        Executer le code suivant dans une nouvelle cellule Google Colab pour créer un fichier .py (appelé streamlit_app.py ici). Puis écrire le script Python associé à l'application Streamlit dans la cellule.
        </pre>

        `%%writefile streamlit_app.py 
        import streamlit as st 
        #Insérer code Python contenant les commandes Streamlit`
        <pre>
        Une fois que le code est terminé, executer le code suivant dans une nouvelle cellule Google Colab pour déployer l'application Streamlit.
        </pre>
        !streamlit run /content/streamlit_app.py & npx localtunnel — port 8501

        <pre>
        Maintenant vous maitrisez Streamlit et ses bonnes pratiques pour tout type de projet et pour tout environnement de travail !

        </pre>
        """)
    
if page == pages[1] : 
    st.write("### WordCloud")
    
    NBWORDS=    st.slider("Choose Number of words",min_value=1,max_value=100,value=10,step=1)
    

    import streamlit as st
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Le tokenizer a déjà tout calculé — word_counts contient les fréquences
    n=100
    word_freq = dict(list(tokenizer.word_counts.items())[:n])

    # Ou plus proprement, trié par fréquence
    word_freq = dict(
        sorted(tokenizer.word_counts.items(), 
            key=lambda x: x[1], 
            reverse=True)[:100]
    )

    @st.cache_data
    def generate_wordcloud(word_freq):
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        return wc

    message = st.chat_input("")

    if message:
        word_freq=print_closest(message,NBWORDS)
        wc = generate_wordcloud(word_freq)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.write(f"you requested {message}",)
        text=st.write(word_freq)
