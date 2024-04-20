import streamlit as st
import os
import pandas as pd
from menu import menu
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# Configuración de la página

PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'



st.set_page_config(page_title="Mi Aplicación Streamlit", page_icon=":rocket:", layout="wide")
st.image(PATH_IMG + "header_ctg.jpg")
menu()

st.subheader("búsqueda de eventos de turismo - Procolombia")
st.divider()
st.markdown("En el aplicativo encontraras información relacionada con eventos asociativos de turismo en Colombia. Se incluyen métodos para extracción y búsqueda de eventos a través de internet, cumpliendo con criterios especificados por el usuario.")
st.markdown("""***¿Que se incluye en el aplicativo?***""")
st.markdown("""***1. Tres metodologías de búsqueda de eventos de interés en internet.***""")
st.markdown("""**- Búsqueda automática:** Por medio de este método, el aplicativo realiza una búsqueda general en internet basado en los criterios especificados en la configuración, posteriormente extrae la información relevante para los eventos encontrados.""")
st.page_link("pages/1-Busqueda_Automatica.py", label="👉 :red[Búsqueda Automatica]")
st.markdown("""**- Búsqueda manual:** Por medio de este método, el usuario puede ingresar una URL y el aplicativo buscara los eventos disponibles en la misma, extrayendo la información relevante.""")
st.page_link("pages/2-Busqueda_Manual.py", label="👉 :red[Búsqueda Manual]")
st.markdown("""**- Búsqueda recursiva:** Por medio de este método, el aplicativo recorrerá diferentes sitios WEB de diferentes entidades en busqueda de eventos, y extrae los que sean encontrados.""")
st.page_link("pages/3-Busqueda Recursiva.py", label="👉 :red[Búsqueda Recursiva]")
st.markdown("""***2. Dashboard:*** El aplicativo cuenta con un panel para seguimiento y visualizacion de los eventos encontrados""")
st.page_link("pages/4-Dashboard.py", label="👉 :red[Dashboard]")
st.markdown("""***3. configuración:*** El aplicativo cuenta con un panel de configuración, donde se incluyen todas las opciones de conexión, almacenamiento y criterios de búsqueda del aplicativo.""")
st.page_link("pages/5-Configuracion.py", label="👉 :red[configuración]")
st.divider()

# Pruebas


from pages.lib.funciones import filtrar_df, cargar_eventos_procesados_archivo, cargar_configuracion, cargar_contraseñas, obtener_criterios_busqueda, actualizar_configuracion
from pages.lib.funciones import limpiar_df_event, web_scrapper, extraer_informacion_general_gemini
from pages.lib.funciones_db import cargar_eventos_procesados_db, check_event_db
from pages.lib.config import FN_KEYW_JSON, ACCESS_PATH 

config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)
contraseñas = cargar_contraseñas(ACCESS_PATH) 


df_eventos_sf = cargar_eventos_procesados_db(contraseñas, 'Snowflake')
df_eventos_mdb = cargar_eventos_procesados_db(contraseñas, 'MongoDB')

st.dataframe(df_eventos_sf)
st.dataframe(df_eventos_mdb)
