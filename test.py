import os, toml, requests
import requests
import datetime as dt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Union
from langchain.utilities import TextRequestsWrapper
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser, YamlOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import traceback

from pages.lib.funciones import cargar_configuracion, cargar_contraseñas, obtener_criterios_busqueda, filtrar_df
from pages.lib.funciones import get_embedding_gemini, extraer_informacion_general_gemini,extraer_informacion_general_gemini_v3,check_event_embedding_gemini 
from pages.lib.funciones import web_scrapper, limpiar_dict_event
from pages.lib.funciones import query_google_search
from pages.lib.funciones_db import mdb_execute_query, cargar_eventos_procesados_db, insert_event_db, insert_errors_db, mdb_actualizar_event_type, insert_google_url_info, update_google_url_info_event
from pages.lib.funciones_db import check_event_db,check_title, check_url 

from pages.lib.config import FN_KEYW_JSON, ACCESS_PATH, PATH_DATA


contraseñas = cargar_contraseñas(ACCESS_PATH)
config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)


criterios_lista = obtener_criterios_busqueda(config)

google_query_result = {1: {'title': 'Congreso Internacional de Ética, Ciencia y Educación',
  'snippet': 'Página · Ciencia, tecnología e ingeniería · +57 301 3714038 · inis.com.co/index.php/congreso-internacional-de -etica-ciencia-y-educacion · Aún sin calificación (0\xa0...',
  'long_description': 'Congreso Internacional de Ética, Ciencia y Educación. 1.351 Me gusta · 8 personas están hablando de esto. V Congreso Internacional de ética, ciencia y educación\nMedellín(Col) 25, 26 y 27 de septi de 2024',
  'link': 'https://m.facebook.com/p/Congreso-Internacional-de-%C3%89tica-Ciencia-y-Educaci%C3%B3n-100063700442592/?locale=es_LA'},
 2: {'title': 'Líneas directrices de la Actividad Internacional - Congreso de los ...',
  'snippet': 'Líneas directrices de la Actividad Internacional - Congreso de los Diputados. Saltar al contenido. Esta web utiliza cookies de terceros y propias para\xa0...',
  'long_description': 'N/A',
  'link': 'https://www.congreso.es/cem/lineas-act-internac'},
 3: {'title': 'Congreso Internacional de Odontología - Universidad del Valle ...',
  'snippet': 'Jun 11, 2015 ... ... Colombia - Código postal: 760042 - Nit: 890.399.010.6. Sede San Fernando, Calle 4B # 36-00 Santiago de Cali, Valle del Cauca, Colombia.',
  'long_description': 'Con la presencia de expertos procedentes de universidades de Europa, Estados Unidos y América\xa0del Sur, se llevará a cabo entre jueves y viernes el Congreso Internacional de Odontología en el\xa0campus San Fernando de la Universidad del Valle. El encuentro organizado por la Escuela de Odontología hace p...',
  'link': 'https://www.univalle.edu.co/proyeccion-internacional/congreso-internacional-odontologia'},
 4: {'title': 'Concluye VI Congreso internacional de Liturgia | Conferencia ...',
  'snippet': 'Aug 4, 2022 ... ... Internacional , Congreso Eucarístico en Quito , conferencia episcopal de colombia , iglesia colombiana · Leer Más. Mar 12 Dic 2023. Están\xa0...',
  'long_description': 'Conferencia Episcopal de Colombia',
  'link': 'https://www.cec.org.co/noticias-de-los-departamentos-del-spec/liturgia/concluye-vi-congreso-internacional-de-liturgia'}}

consulta = None
coleccion = 'fct_eventos'
df = mdb_execute_query(consulta,coleccion, contraseñas['mongo_db'])
df_events_hist_filter = df[(df['status'] == "OK") &
                                        (df['there_is_event'] == True) &
                                        ((df['year_parsed'] >= dt.datetime.today().year-10) | (df['year_parsed'].isna()))]
i=1
google_query_result={}
for row in df_events_hist_filter.iterrows():
    dict_google = {
        'title':row[1].google_title,
        'link':row[1].google_url,
    }
    google_query_result[i] =  dict_google
    i+=1



date =  dt.datetime.today().date().strftime("%Y-%m-%d")
stats = {'urls':0, 'urls_eventos':0, 'urls_eventos_nuevos':0, 'eventos' : 0}
df_events_busqueda = pd.DataFrame()
df_errores_busqueda = pd.DataFrame()
key_W = "(Congreso)(Internacional)"
for item in google_query_result.keys():
    stats['urls'] += 1
    url = google_query_result[item]['link']
    title = google_query_result[item]['title']
    print("###############################################################")
    print(url)
    
    #bar.progress(i+step)
    #i = i+step
    #static_1.markdown('**Criterio:** {}'.format(key_W['exactTerms']))
    #static_2.markdown('**Link**: {}'.format(url))
    #static_3.markdown('**Progreso:** {} %'.format(round(i*100,0)))
    if (check_url(url, contraseñas, config['base_datos'])):
            
        print("URL Ya Procesado")
        continue
    else:
        try:
            print("URL No Procesado")
            event_val_result, event_info_list,tokens_size, context_words  = extraer_informacion_general_gemini_v3(url, contraseñas["api_gemini"]['KEY'])
            
            
            if (event_val_result.there_is_event == True or event_val_result.there_is_event == 'True') and  len(event_info_list.events) > 0 :
                
                stats['urls_eventos'] += 1
                if event_info_list != None:
                    stats['eventos'] += 1
                    for event in event_info_list.events:
                        if event.there_is_event == "True" and event.title != None:
                            print("Evento encontrado: {}".format(event.title))
                            if(check_title(event.title, contraseñas, config['base_datos'])):
                                print("Evento ya encontrado por titulo")
                            else:
                                print("Evento no procesado segun titulo")
                                
                                if(check_event_embedding_gemini(event, contraseñas)):
                                    print("Evento ya encontrado por busqueda semantica")
                                else:
                                    print(f"Evento no procesado segun Busqueda Semantica, Contexto {context_words}, tokens {tokens_size}") 
                                    event_text = f"{event.title}, {event.description},  {event.date}, {event.year}, {event.country}, {event.city}"   
                                    event = event.__dict__
                                    event['url'] = url
                                    event['embedding'] = get_embedding_gemini(str(event_text), contraseñas["api_gemini"]['KEY'])
                                    event['date_processed'] =  dt.datetime.today()
                                    event['tokens_size'] = tokens_size
                                    event['context_words'] = context_words
                                    event = limpiar_dict_event(event)
                                    resultado = insert_event_db([event], contraseñas, config['base_datos'])
                                    if resultado == True:
                                        print("Evento Insertados Correctamente")
                                    else:
                                        print("Error Insertando Evento. Error: {}".format(resultado))
                else: 
                    print(event_info_list)
            else:
                print (f"No Event: {event_val_result.there_is_event}")
                
            df_google_info = pd.DataFrame([google_query_result[item]])
            df_google_info = df_google_info.rename(columns={'title':'google_title',
                                                            'snippet':'google_snippet',
                                                            'long_description': 'google_long_description',
                                                            'link':'google_url'})
            df_google_info['_id'] = url
            df_google_info['criterio'] = key_W
            google_info = df_google_info.to_dict(orient='records')
            insert_google_url_info(google_info, contraseñas, config['base_datos'])
            
        except Exception as e:
            traceback.print_exc()
            dict_error = {
                'status': 'ERROR',
                'error': str(e),
                'date_processed' : date,
                'google_url': url
            }
            print(f"Error:{e}" )
            resultado = insert_errors_db(dict_error, contraseñas, config['base_datos'])  
            if resultado == True:
                print("Errores Insertados Correctamente")
            else:
                print("Error Insertando Evento. Error: {}".format(resultado))  
        
