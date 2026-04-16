# uv - useful or useless ?

    So in our progress to use uv and manage multiple environments let's consider the following scenario 
    - let say we need to test an alternate env, while keeping something working and willing to try a litle alternative.

# summary of combinations
    `uv env .bla`
    `source .bla/Scripts/activate`
    `uv sync`            ## may populate .venv (creating it in the background)
    `uv sync --active`   ## might be dues if missconf bellow
    `uv add foo`
    `uv add foo --active`

## Explanations
if you run *uv venv .bla*, you inherit a *.bla/Scripts/activate* that is hardcoded toward *.bla*, but not only,
also toward its parent folder, preventing move and renaming of the parent folder with ease !!
So what do you need to know if you wish to relocate you projet, case of a redeployment, this is what
we explore here.

if you rename your projet (for any conflict reason) or move it
the activate script becomes obsolete, pointing to the old name and location (bouh)

So one suggestion is to go and update the scripts - but what a hell stupide to modify generated script by hand - at any time you loose your configs

And so yep, you can `rm -rf .venv` because to retrieve the promise is that you simply need a `uv venv .bla` and a `uv sync`
Well, not exaclty.

litle problem as uv is not taking into accout ut UV_VIRTUALENV defined, but goes to kind of hardcoded .venv - over and over

/!\ Hence, despite a proper .bla activation, a first uv sync will sync to .venv silently !!!

All next uv sync will complain that the active .bla is not matching the project environment path

    **warning**: `VIRTUAL_ENV=.bla` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead

So when running `uv venv .bla`, this will create the .bla env with the new parent folder location
"uv sync" this will rename

IN ORDER TO KEEP uv sync TO the active env, there is two configurations to address

The first one in .env file with 
## .env
`export UV_PROJECT_ENVIRONMENT=.bla`

As a consequence, in a step you have to switch betweed env, it can be that the pyproject.toml might be impacted, the uv.lock as well - so it really depends on the attempt, it might be good to backup the toml

add this .env also to the .gitignore

second file to configure is the .vscode/settings.json, to allow vscode to use the .env injection
## vscode/settings.json
{
    "python.terminal.useEnvFile": true,
    "python.envFile": "${workspaceFolder}/.env"
}


as most possible use

So it is, please take care !


Next topic is about streamlit

Streamlit architecture is simple and terric and real strategy for what can be provide must be checked with major care

Where to calculate things:
❌ Main script level        → recalculated at EVERY interaction
✅ @st.cache_data           → result memoized by args (data)
✅ @st.cache_resource       → shared singleton (ML model, DB connection)
✅ st.session_state         → state that must persist between runs
✅ @st.fragment             → isolate a subtree from the global rerun

The golden rule: what is slow must be behind a cache or in session_state


so we will try to see what kind of architecture we can get


## Deployment ?

Several questions will araise for deployment and sharing
we have a platform to reach and push a whole object composed on data and methods
on the one hand the .py program, on the other hand the 

streamlit to start the server locally
`streamlit run your_script.py`

elemetary architecture to keep in mind
picts\streamlit_project_architecture.svg

## The golden rule of data in Streamlit
Your repo = code only. Data lives elsewhere and is fetched at runtime. The repo is cloned by Streamlit Cloud on every deploy — it's not a data store.

Project structure (what goes in your repo)
my-app/
├── app.py                    # main entry point
├── pages/
│   ├── 01_dashboard.py
│   └── 02_analysis.py
├── src/
│   ├── data_loader.py        # all data-fetching logic
│   ├── charts.py
│   └── utils.py
├── data/
│   └── reference.csv         # ONLY tiny static files (< a few MB)
├── .streamlit/
│   ├── config.toml           # theme, server settings
│   └── secrets.toml          # LOCAL ONLY — never commit this
├── requirements.txt
└── .gitignore                # must include: .env, secrets.toml, *.pkl, data/raw/
The key discipline: src/data_loader.py calls external services. The repo never stores the actual data.



How Streamlit Cloud deployment works

1 You connect your GitHub repo in the Streamlit Cloud UI (one click)
2 Every git push to your chosen branch auto-redeploys the app
3 Streamlit Cloud clones the repo, runs pip install -r requirements.txt, then launches streamlit run app.py
4 The filesystem is ephemeral — anything written to disk disappears on restart

So credentials for your data sources are never in the repo — they go into the Secrets panel in the Streamlit Cloud UI, then accessed in code via st.secrets["my_key"].

The data strategy — choosing the right backend
Your data                     Right tool                 How
A few CSVs, < 50 MB           Commit to repo             pd.read_csv("data/file.csv")
Medium files, rarely updated  GitHub Releases or S3      boto3 + @st.cache_data
Structured, queryable         Postgres / Supabase        st.connection("postgresql")
Live, changes often           API/Snowflake/BigQuery     native connectors
ML models / large datasets    Hugging Face Hub or DVC+S3 huggingface_hub.hf_hub_download()

# src/data_loader.py
import streamlit as st
import pandas as pd
import boto3

@st.cache_data(ttl=3600)          # re-fetches every hour, cached in between
def load_sales_data():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["aws_key"],
        aws_secret_access_key=st.secrets["aws_secret"],
    )
    obj = s3.get_object(Bucket="my-bucket", Key="data/sales.parquet")
    return pd.read_parquet(obj["Body"])

The @st.cache_data decorator means the S3 call runs once per hour, not on every re-run. That's the critical coupling between external data and Streamlit's execution model.



Secrets — the right way
Locally — create .streamlit/secrets.toml (never committed):
toml[aws]
key = "AKIA..."
secret = "xyz..."

[postgresql]
url = "postgresql://user:pass@host/db"
On Streamlit Cloud — paste the same content in Settings → Secrets. That's it — no CI/CD pipeline needed, no environment variable gymnastics.


The most common patterns in the wild
Small project / demo → a couple of CSVs committed to the repo, done.

Real dashboard with live data → Supabase (free tier) + st.connection() — the easiest path to a proper database without infrastructure.

ML / data science app → data and model weights on Hugging Face Hub (free, public), loaded at startup behind @st.cache_resource.

Enterprise → S3 or BigQuery, credentials in Streamlit Cloud secrets, DVC for data versioning alongside the code.

Want to go deeper on any of these patterns — the st.connection() API, DVC workflow, or how to structure a multi-page app properly?

