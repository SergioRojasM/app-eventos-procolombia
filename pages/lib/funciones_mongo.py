from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import numpy as np

#### FUNCIONES MONGODB

def mdb_insert_doc(df, nombre_coleccion, mdb_config):
    uri = f"mongodb+srv://{mdb_config['user']}:{mdb_config['password']}@{mdb_config['cluster']}.hscob2f.mongodb.net/?retryWrites=true&w=majority&appName={mdb_config['cluster']}"
    conn_status = False
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client[mdb_config['database']]
        coleccion = db[nombre_coleccion]
        client.admin.command('ping')
        conn_status = True
    except:
        print("Error Conectando a MongoDB")
    
    if conn_status:
        try:
            documento = df.to_dict(orient='records')
            coleccion.insert_many(documento)
            return (True)
        except Exception as e:
            return('Error Cargando la informacion. Error: {}'.format(e))
        
def mdb_execute_query(consulta, nombre_coleccion, mdb_config):
    uri = f"mongodb+srv://{mdb_config['user']}:{mdb_config['password']}@{mdb_config['cluster']}.hscob2f.mongodb.net/?retryWrites=true&w=majority&appName={mdb_config['cluster']}"
    conn_status = False
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client[mdb_config['database']]
        coleccion = db[nombre_coleccion]
        client.admin.command('ping')
        conn_status = True
    except:
        print("Error Conectando a MongoDB")
    
    if conn_status:
        try:
            documentos = list(coleccion.find(consulta))
            df = pd.DataFrame(documentos)
            return (df)
        except Exception as e:
            print ( f"Error:{e}")
            return('Error Cargando la informacion. Error: {}'.format(e))
        
def mdb_cargar_eventos_procesados_db(mdb_config):
    consulta = None
    coleccion = 'fct_eventos'
    return mdb_execute_query(consulta,coleccion, mdb_config)

def mdb_check_event_db(google_url, google_title, mdb_config):
    consulta = {"google_url": google_url}
    coleccion = 'fct_eventos'
    if  len(mdb_execute_query(consulta,coleccion, mdb_config)) > 0:
        return True
    else:
        return False


    