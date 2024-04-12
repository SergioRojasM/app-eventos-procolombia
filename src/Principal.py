import streamlit as st
import os
import pandas as pd
from menu import menu
# Configuración de la página

PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"



st.set_page_config(page_title="Mi Aplicación Streamlit", page_icon=":rocket:", layout="wide")
st.image(PATH_DATA + "header_ctg.jpg")
menu()

eventos_keyw = pd.read_csv(PATH_DATA + "db_eventos_keyw.csv")
st.write(eventos_keyw)

