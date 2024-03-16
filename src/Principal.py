import streamlit as st
import os
import pandas as pd
# Configuración de la página

CWD_PATH = os.getcwd()
DATA_PATH = CWD_PATH + "/src/data/"

eventos_keyw = pd.read_csv(DATA_PATH + "db_eventos_keyw.csv")
st.set_page_config(page_title="Mi Aplicación Streamlit", page_icon=":rocket:")

st.write(eventos_keyw)
