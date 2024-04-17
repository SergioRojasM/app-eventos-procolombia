import streamlit as st
import datetime as dt
import os, toml, requests
import pandas as pd
from pages.lib.funciones import cargar_eventos_procesados_archivo, filtrar_df, cargar_contraseñas
from pages.lib.funciones_mongo import mdb_cargar_eventos_procesados_db
from pages.lib.funciones_snowflake import sf_cargar_eventos_procesados_db
from menu import menu
import plotly.express as px

# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'
FN_KEYW = 'db_eventos_keyw.csv'
FN_EVENTS = 'events_data.xlsx'
FN_KEYW_JSON = 'app_config.json'
ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"
#


menu()
st.image(PATH_IMG + "header_rio.jpg")
