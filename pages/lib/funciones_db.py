from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import snowflake.connector
import pandas as pd
import numpy as np
from datetime import datetime


def mdb_insert_doc(documento, nombre_coleccion, mdb_config):
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
            #documento = df.to_dict(orient='records')
            coleccion.insert_many(documento)
            return (True)
        except Exception as e:
            return('Error Cargando la informacion. Error: {}'.format(e))
        
def mdb_update_doc(doc_id, df, nombre_coleccion, mdb_config):
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
            # actualizacion = { "$set": { 'events':  documento} }
            actualizacion = { "$push": { 'events': { "$each": documento } } }
            # Actualizar el documento
            resultado = coleccion.update_one({ "_id": doc_id }, actualizacion)
            
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
    coleccion = 'fct_eventos_turismo'
    df = mdb_execute_query(consulta,coleccion, mdb_config)
    #df = df.drop(columns='_id')
    return df

def mdb_check_event_db(google_url, google_title, mdb_config):
    consulta = {"google_url": google_url}
    coleccion = 'google_url_info'
    if  len(mdb_execute_query(consulta,coleccion, mdb_config)) > 0:
        return True
    else:
        return False
    
def mdb_check_url(url, mdb_config, coleccion):
    consulta = {"google_url": url}
    coleccion = coleccion
    if  len(mdb_execute_query(consulta,coleccion, mdb_config)) > 0:
        return True
    else:
        return False
    
def mdb_check_title(title, mdb_config, coleccion):
    consulta = {"title": title}
    coleccion = coleccion
    if  len(mdb_execute_query(consulta,coleccion, mdb_config)) > 0:
        return True
    else:
        return False
    
def mdb_actualizar_event_type(event_type,id_evento, nombre_coleccion, mdb_config):
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
            filtro = {'_id': id_evento}  # Suponiendo que el identificador del evento es '_id'
            nuevo_valor = {'$set': {'event_type': event_type}}
            resultado = coleccion.update_one(filtro, nuevo_valor)
            return(f'Cargado Correctamente. event_type{event_type} resultado:{resultado}')
        except Exception as e:
            print ( f"Error:{e}")
            return('Error Cargando event_type. Error: {}'.format(e))
#### FUNCIONES SNOWFLAKE

def sf_insert_rows(df, nombre_tabla, sf_config):
    conn_status = False
    try:
        conn = snowflake.connector.connect(
            user=sf_config['user'],
            password=sf_config['password'],
            account=sf_config['account'],
            warehouse=sf_config['warehouse'],
            database=sf_config['database'],
            schema=sf_config['schema']
        )
        cur = conn.cursor()
        conn_status = True
    except:
        return('Error en conexion a la base de datos')
    
    if conn_status:
        #try:
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna('')
                df[column] = df[column].replace( 'None', None)
            if df[column].dtype == 'float64':
                df[column] = df[column].replace(np.nan, None)
        for index, row in df.iterrows():
            print(df)
            # Construir la consulta SQL dinámicamente
            columns = ', '.join(row.index)
            placeholders = ', '.join(['%s'] * len(row))
            query = f"INSERT INTO {nombre_tabla} ({columns}) VALUES ({placeholders})"
            row_values = [str(value) if isinstance(value, datetime) else value for value in row] 
            # Ejecutar la consulta con los valores de la fila actual
            cur.execute(query, tuple(row_values))

        # Confirmar la transacción y cerrar la conexión
        conn.commit()
        cur.close()
        conn.close()
        return (True)
        #except Exception as e:
        #    return('Error Cargando la informacion. Error: {}'.format(e))

def sf_execute_query(sql, sf_config):
    conn_status = False
    try:
        conn = snowflake.connector.connect(
            user=sf_config['user'],
            password=sf_config['password'],
            account=sf_config['account'],
            warehouse=sf_config['warehouse'],
            database=sf_config['database'],
            schema=sf_config['schema']
        )
        cur = conn.cursor()
        conn_status = True
    except:
        return(e)
    
    if conn_status:
        try:
            cur.execute(sql)
            df_read = cur.fetch_pandas_all()
            conn.close()   
            cur.close()
            conn.close()
            return (df_read)
        except Exception as e:
            return(e)

def sf_cargar_eventos_procesados_db(sf_config):
    sql = "SELECT * FROM BODEGA_DWH.FCT_EVENTOS"
    df = sf_execute_query(sql, sf_config)
    df.columns = [s.lower() for s in df.columns] 
    return df
        
def sf_check_event_db(google_url, google_title, sf_config):
    sql = f"SELECT * FROM BODEGA_DWH.FCT_EVENTOS WHERE GOOGLE_URL='{google_url}' OR GOOGLE_TITLE='{google_title}'"
    if  len(sf_execute_query(sql, sf_config)) > 0:
        return True
    else:
        return False
    
    
# HANDLER GENERAL


def cargar_eventos_procesados_db(config, db):
    if db == "MongoDB":
        return mdb_cargar_eventos_procesados_db(config['mongo_db'])
    elif db == "Snowflake":
        return sf_cargar_eventos_procesados_db(config['snowflake'])
    
def check_event_db(google_url, google_title, config, db):

    if db == "MongoDB":
        return mdb_check_event_db(google_url, google_title, config['mongo_db'])
    elif db == "Snowflake":
        return sf_check_event_db(google_url, google_title, config['snowflake'])

def check_url(url, config, db):
    destino = 'urls_procesadas'
    if db == "MongoDB":
        return mdb_check_url(url, config['mongo_db'], destino)
    elif db == "Snowflake":
        return sf_check_event_db(url, config['snowflake'])
    
def check_title(title, config, db):
    destino = 'fct_eventos_turismo'
    if db == "MongoDB":
        return mdb_check_title(title, config['mongo_db'], destino)

    
def insert_event_db(doc, config, db):
    destino = 'fct_eventos_turismo'
    if db == "MongoDB":
        return mdb_insert_doc(doc, destino, config['mongo_db'])
    elif db == "Snowflake":
        return sf_insert_rows(doc, destino, config['snowflake'])
    
def insert_errors_db(doc, config, db):
    destino = 'fct_errores'
    if db == "MongoDB":
        return mdb_insert_doc(doc, destino, config['mongo_db'])
    elif db == "Snowflake":
        return sf_insert_rows(doc, destino, config['snowflake'])
    
def insert_google_url_info(df, config, db):
    destino = 'urls_procesadas'
    if db == "MongoDB":
        return mdb_insert_doc(df, destino, config['mongo_db'])
    elif db == "Snowflake":
        return sf_insert_rows(df, destino, config['snowflake'])
    
def update_google_url_info_event(id, df, config, db):
    if db == "MongoDB":
        destino = 'google_url_info'
        return mdb_update_doc(id, df, destino, config['mongo_db'])
    elif db == "Snowflake":
        destino = 'google_url_info'
        # return sf_insert_rows(df, destino, config['snowflake'])
    
def mdb_get_k_nearest_results(embedding, k, nombre_coleccion, mdb_config):

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
        return False
        
    if conn_status:
        pipeline = [
        {
            '$vectorSearch': {
            'index': 'event_vector_index', 
            'path': 'embedding', 
            'queryVector': embedding,
            'numCandidates': 20, 
            'limit': k
            }
        }
        ]
        result = coleccion.aggregate(pipeline)
        return result
    else:
        return False    
        