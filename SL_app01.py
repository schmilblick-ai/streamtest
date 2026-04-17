import streamlit as st
from backend.utils import load_css


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

##############################################################################################
# Create a sidebar selection
##############################################################################################
main        = st.Page("pages/0_main.py", title="main", icon=":material/dashboard:")
st.sidebar.page_link(main, label='🏠Home',)


## central various header definition - for incubation of various proposals
def cinoch_header():
    st.title("Word2Vec Explorer - Avis cinéma")
    st.caption("Exploration des embeddings entraînés sur une base d'avis de films")   

def main_header():
    st.title("Explore TrustPilot multilabel classifier")
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


if True:

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


    pg = st.navigation({"main":[main]
        ,"🎬Cinéma": [mot_proches, analogie, clustering, outliers]
        ,"⚙️System": [osexplo]
        ,"🧑‍✈️TrustP": [Marvin, Lionel, Robin, Whole]
        },)
    
    main_header()
    pg.run()

    

if False:
    st.write(f"where are we {dir(pg._page)} {pg._page}")
    st.write(f"where are we {pg._page.parent},{pg._page} {outliers} {dir(outliers)} {outliers._url_path}")
    ##############################################################################################

