print('inicio')
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from pages.lib.funciones import cargar_contraseñas
from pages.lib.config import FN_KEYW_JSON, ACCESS_PATH, PATH_DATA
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
contraseñas = cargar_contraseñas(ACCESS_PATH)

from pages.lib.funciones import comparar_eventos_gemini

def get_embedding_gemini(text, API_KEY_GEMINI ):
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="semantic_similarity")

    vector = embeddings.embed_query(text)
    return vector

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

def check_event_embedding_gemini(event_in, contraseñas):
    query = 'Congreso Internacional de Odontología'

    embedding = get_embedding_gemini(query, contraseñas["api_gemini"]['KEY'])
    k_events  = mdb_get_k_nearest_results(embedding, 5, 'fct_eventos_turismo', contraseñas["mongo_db"])
    flag_event = False

    if k_events:
        for event_db in k_events:
            event_db_text = f"{event_db['title']}, {event_db['description']},  {event_db['date']}, {event_db['year']}, {event_db['country']}, {event_db['city']}"
            llm_result = comparar_eventos_gemini(event_in, event_db_text, contraseñas["api_gemini"]['KEY'])
            print(event_in)
            print(event_db_text)
            print(llm_result)
            if llm_result.are_same_event>90:
                flag_event = True
            
    return flag_event
                
                


print('Inicio query LLM')
event = """'there_is_event': 'True',
            'event_type': 'Congress',
            'title': 'Congreso Internacional de Odontología',
            'general_title': 'Congreso Internacional de Odontología',
            'date': '11 y 12 de junio',
            'year': '2015',
            'description': 'Con la presencia de expertos procedentes de universidades de Europa, Estados Unidos y América del Sur, se llevará a cabo entre jueves y viernes el Congreso Internacional de Odontología en el\xa0campus San Fernando de la Universidad del Valle.',
            'country': 'Colombia',
            'city': 'Cali',
            'place': 'campus San Fernando de la Universidad del Valle',
            'key_words': 'Odontología, Salud Bucal, Caries, Ortodoncia, Implantología',
            'asistants': 'None',
            'event_category': 'Medical Sciences',
            'url': 'https://www.univalle.edu.co/proyeccion-internacional/congreso-internacional-odontologia'
            """
ref_event ="""
            'title': 'Congreso Internacional de Odontología',
            'description': 'Con la presencia de expertos procedentes de universidades de Europa, Estados Unidos y América del Sur, se llevará a cabo entre jueves y viernes el Congreso Internacional de Odontología en el\xa0campus San Fernando de la Universidad del Valle.',
            'date': '11 y 12 de junio',
            'year': '2015',
            'country': 'Colombia',
            'city': 'Cali',
            """
print(check_event_embedding_gemini(ref_event, contraseñas))





    