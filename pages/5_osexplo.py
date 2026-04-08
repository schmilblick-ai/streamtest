
import subprocess
import streamlit as st

cmd = st.text_input(
    "commands to execute",
    value="ls -arl; pwd"
)

answ = st.empty()
answ.text_area(label="output")
#answ.markdown("tst")
if cmd:
   # output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    


    #output = str(output).replace('\\r\\n','  \\n\\n')
    #output="<pre>\"+"b"

    answ.text_area(label="output",value=output,)
    #answ.markdown(output)