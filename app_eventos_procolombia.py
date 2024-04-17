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

st.subheader("Busqueda de eventos de turismo - Procolombia")
st.divider()
st.markdown("En el aplicativo encontraras informacion relacionada con eventos asociativos de turismo en Colombia. Se incluyen metodos para extraccion y busqueda de eventos a traves de internet, cumpliendo con criterios especificados por el usuario.")
st.markdown("""***¿Que se incluye en el aplicativo?***""")
st.markdown("""***1. Tres metodologias de busqueda de eventos de interes en internet.***""")
st.markdown("""**- Busqueda automatica:** Por medio de este metodo, el aplicativo realiza una busqueda general en internet basado en los criterios especificados en la configuracion, posteriormente extrae la informacion relevante para los eventos encontrados.""")
st.page_link("pages/1-Busqueda_Automatica.py", label="👉 :red[Busqueda Automatica]")
st.markdown("""**- Busqueda manual:** Por medio de este metodo, el usuario puede ingresar una URL y el aplicativo buscara los eventos disponibles en la misma, extrayendo la informacion relevante.""")
st.page_link("pages/2-Busqueda_Manual.py", label="👉 :red[Busqueda Manual]")
st.markdown("""**- Busqueda recursiva:** Por medio de este metodo, el aplicativo recorrera diferentes sitios WEB de diferentes entidades en busqueda de eventos, y extrae los que sean encontrados.""")
st.page_link("pages/3-Busqueda Recursiva.py", label="👉 :red[Busqueda Recursiva]")
st.markdown("""***2. Dashboard:*** El aplicativo cuenta con un panel para seguimiento y visualizacion de los eventos encpntrados""")
st.page_link("pages/4-Dashboard.py", label="👉 :red[Dashboard]")
st.markdown("""***3. Configuracion:*** El aplicativo cuenta con un panel de configuracion, donde se incluyen todas las opciones de conexion, almacenamiento y criterios de busqueda del aplicativo.""")
st.page_link("pages/5-Configuracion.py", label="👉 :red[Configuracion]")

 