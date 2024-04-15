import snowflake.connector
import pandas as pd
import numpy as np
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
        try:
            for column in df.columns:
                if df[column].dtype == 'object':
                    df[column] = df[column].fillna('')
                if df[column].dtype == 'float64':
                    df[column] = df[column].replace(np.nan, None)
                    
            for index, row in df.iterrows():
                print(index)
                # Construir la consulta SQL dinámicamente
                columns = ', '.join(row.index)
                placeholders = ', '.join(['%s'] * len(row))
                query = f"INSERT INTO {nombre_tabla} ({columns}) VALUES ({placeholders})"
                
                # Ejecutar la consulta con los valores de la fila actual
                cur.execute(query, tuple(row))

            # Confirmar la transacción y cerrar la conexión
            conn.commit()
            cur.close()
            conn.close()
            return (True)
        except Exception as e:
            return('Error Cargando la informacion. Error: {}'.format(e))

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
    return sf_execute_query(sql, sf_config)
        
def sf_check_event_db(google_url, google_title, sf_config):
    sql = f"SELECT * FROM BODEGA_DWH.FCT_EVENTOS WHERE GOOGLE_URL='{google_url}' OR GOOGLE_TITLE='{google_title}'"
    if  len(sf_execute_query(sql, sf_config)) > 0:
        return True
    else:
        return False