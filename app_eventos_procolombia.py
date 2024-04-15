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

mdb_config = st.secrets['mongo_db']
st.write(mdb_config)
uri = f"mongodb+srv://{mdb_config['user']}:{mdb_config['password']}@{mdb_config['cluster']}.hscob2f.mongodb.net/?retryWrites=true&w=majority&appName={mdb_config['cluster']}"
conn_status = False
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[mdb_config['database']]
    coleccion = db['fct_events']
    client.admin.command('ping')
    conn_status = True
except Exception as e:
    st.write(f"Error Conectando a MongoDB: {e}")

if conn_status:
    try:
        documentos = list(coleccion.find())
        df = pd.DataFrame(documentos)
        st.dataframe(df)
    except Exception as e:
        st.write ( f"Error:{e}")


 