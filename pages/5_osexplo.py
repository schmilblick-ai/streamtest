
import subprocess
import streamlit as st
from SL_app01 import osexplo_header

osexplo_header()
if False:
    cmd = st.text_input("OS commands to execute",value="")

    answ = st.empty()
    answ.text_area(label="output",key="output_answ00")
    #answ.markdown("tst")
    output=""
    if cmd:
    # output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
        output =subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
        nblines = len(output.splitlines())
        answ.text_area(label="output",value=output,key="output_answ00",height=nblines*20)


# 1. Initialiser le state UNE SEULE FOIS
if False:

    if "history" not in st.session_state:
        st.session_state.history = []

    # Barre supérieure : titre + toggle clear
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### Terminal")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()


    # 2. Afficher l'historique (calculé avant l'input)
    for entry in st.session_state.history:
        st.text(f"$ {entry['cmd']}")
        st.code(entry['output'], language="bash")

    if False:
        # 3. Input avec on_change ou bouton — la valeur est préservée
        cmd = st.text_input("Prompt", key="cmd_input", placeholder="ls -la", value="")

        # 4. Exécution via bouton (déclenche un re-run explicite)
        if st.button("Exécuter") and cmd:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True #, cwd="/home"
            )
            output = result.stdout or result.stderr
            st.session_state.history.append({"cmd": cmd, "output": output})
            st.rerun()  # Force un re-run immédiat pour rafraîchir l'affichage

    def execute_cmd():
        cmd = st.session_state.cmd_input
        if cmd:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            st.session_state.history.append({
                "cmd": cmd,
                "output": result.stdout or result.stderr
            })
            # Reset l'input après exécution
            st.session_state.cmd_input = ""

    st.text_input("Commande", key="cmd_input", on_change=execute_cmd)


if "history" not in st.session_state:
    st.session_state.history = []

def execute_cmd():
    cmd = st.session_state.cmd_input
    if cmd:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.session_state.history.append({
            "cmd": cmd,
            "output": result.stdout or result.stderr
        })
        st.session_state.cmd_input = ""

# Historique
for entry in st.session_state.history:
    st.text(f"$ {entry['cmd']}")
    st.code(entry['output'], language="bash")

# Input
st.text_input("Commande", key="cmd_input", placeholder="ls -la", on_change=execute_cmd)

#Si tu veux le bouton encore plus discret, tu peux le styler avec du markdown ou réduire sa largeur via st.columns :
python_, col = st.columns([5, 1])
with col:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if False:
    """
    df -Ph ~; pwd ~; whoami ; grep $(whoami) /etc/passwd
    Filesystem      Size  Used Avail Use% Mounted on
    overlay         248G  131G  118G  53% /
    /mount/src/streamtest
    appuser
    appuser:x:2000:2000::/home/appuser:/bin/bash

    """    