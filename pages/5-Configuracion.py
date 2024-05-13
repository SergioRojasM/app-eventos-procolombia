import streamlit as st
import datetime as dt
import os, toml, requests
import pandas as pd
from pages.lib.funciones import cargar_eventos_procesados_archivo, filtrar_df, cargar_contraseñas

from menu import menu
import plotly.express as px
from pages.lib.funciones import cargar_configuracion,  actualizar_configuracion


# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'
FN_KEYW = 'db_eventos_keyw.csv'
FN_EVENTS = 'events_data.xlsx'
FN_KEYW_JSON = 'app_config.json'
ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"

MODELS_DICT = {'Gemini':0, 'GROQ-LLAMA2':1}
DB_DICT = {'MongoDB':0, 'Snowflake':1}
PERIODO_DICT = {"Sin restriccion" : 0, "Ultimo año":1, "Ultimo mes":2, "Ultima semana":3}
ORDEN_DICT = {"Sin orden":0, "Mas Recientes":1, "Los dos metodos":2}

#


menu()
st.image(PATH_IMG + "header_rio.jpg")

config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)

st.header("Configuracion por defecto")
col1_conf, col2_conf = st.columns([4,4])
col1_conf.markdown("***Modelo LLM*** ")
radio_modelo = col1_conf.radio(
                        "Seleccione un modelo 👉",
                        key="model",
                        options=["Gemini", "GROQ-LLAMA2"],
                        index= MODELS_DICT[config['modelo']],
                        horizontal = True
                    )

col1_conf.markdown("***Base de datos*** ")
config['base_datos'] = col1_conf.radio(
                "Seleccionar Base de datos 👉",
                key="db",
                options=["MongoDB", "Snowflake"],
                index= DB_DICT[config['base_datos']],
                horizontal = True
            )

bot_act_conf =  col2_conf.button("Actualizar Configuracion")
if bot_act_conf:    
    actualizar_configuracion(config)
    st.rerun()

