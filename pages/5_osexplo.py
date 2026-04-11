
import subprocess
import streamlit as st

cmd = st.text_input("OS commands to execute",value="")

answ = st.empty()
answ.text_area(label="output")
#answ.markdown("tst")
if cmd:
   # output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    answ.text_area(label="output",value=output)

"""
df -Ph ~; pwd ~; whoami ; grep $(whoami) /etc/passwd
Filesystem      Size  Used Avail Use% Mounted on
overlay         248G  131G  118G  53% /
/mount/src/streamtest
appuser
appuser:x:2000:2000::/home/appuser:/bin/bash

"""    