
import streamlit as st
import os, toml, requests
import requests
import datetime as dt
import pandas as pd
import nltk, json
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
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
import numpy as np
import google.generativeai as genai
from menu import menu

from pages.lib.funciones import filtrar_df, cargar_eventos_procesados_archivo, cargar_configuracion, cargar_contraseñas, obtener_criterios_busqueda, actualizar_configuracion
from pages.lib.funciones import limpiar_df_event, web_scrapper, extraer_informacion_general_gemini
#from pages.lib.funciones_snowflake import sf_cargar_eventos_procesados_db, sf_check_event_db, sf_insert_rows
#from pages.lib.funciones_mongo import mdb_cargar_eventos_procesados_db, mdb_check_event_db, mdb_insert_doc
from pages.lib.funciones_db import cargar_eventos_procesados_db, check_event_db, insert_event_db, insert_errors_db

# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'

FN_KEYW = 'db_eventos_keyw.xlsx'
FN_EVENTS = 'events_data.xlsx'
FN_ERRORS= 'events_error.xlsx'
FN_EVENTS_TODAY = 'events_data_today.xlsx'
FN_EVENTS_FILTER = 'events_data_filter.xlsx'
FN_KEYW_JSON = 'app_config.json'
ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"
#

                        
pd.set_option('future.no_silent_downcasting', True)
# Configuracion de la pagina
st.set_page_config(page_title="Busqueda Automatica", page_icon=":rocket:",layout="wide")
st.image(PATH_IMG + "header_verde.jpg")
st.subheader("Busqueda de Eventos de Turismo")
menu()
tab1, tab2= st.tabs(["Busqueda Automatica", "Resultados"])
tab1_col1, tab1_col2 = tab1.columns([2, 5])
static_0 = tab1_col2.empty()
static_1 = tab1_col2.empty()
static_2 = tab1_col2.empty()
static_3 = tab1_col2.empty()
# Define your desired data structure.

class event(BaseModel):
    # resume: str = Field(description="The resume of the context in few words")
    there_is_event: str = Field(description="Defines if any asociative event is mentioned. If so answer 'Yes', if not answer 'No'")
    title: Optional[str] = Field(description="The name of the event, dont use Acronyms, dont use colon punctuation")
    general_title: Optional[str] = Field(description="The name of the event, dont use Acronyms, don't use colon punctuation, don't specify the version of the event")
    date: Optional[str] = None 
    year: Optional[str] = Field(description="The year of the event, only one year. if not sure use None")
    description: Optional[str] = Field(description="Resumen Corto del evento sin signos de puntuacion. ")
    country: Optional[str] = Field(description="The location of the event, if not sure use None")
    city: Optional[str] = Field(description="The city of the event, if not sure use None")
    place: Optional[str] = Field(description="The name of the place where the event takes place, if not sure use None")
    key_words: Optional[str] = Field(description="Only five key words of thats describe de event, separated by comma, if not sure use None")
    asistants: Optional[str] = Field(description="Information about number of asistants to the event, if not sure use None")
    
class Event(BaseModel):
    title: str  = Field(description="The name of the event, dont use initials, dont use punctuation marks")
    year: Optional[str]   = Field(description="The year of the event")
    country: Optional[str] = Field(description="The location of the event")

class json_resp_events(BaseModel):
    events: List[Event] = Field(..., description="The Event details")

class eventAsist(BaseModel):
    title: str  = Field(description="The name of the event, dont use initials, dont use punctuation marks")
    participants: Optional[str]   = Field(description="The resume of the information in few words about event participation, if not information or you are not sure put None")

def cargar_llm(GEMINI_API):

    os.environ["GOOGLE_API_KEY"] = GEMINI_API
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    return llm

def query_google_search(page=1, search_engine_keys=None, add_params = {}):
  """
  Query the Google Custom Search API and return the results in a dictionary.

  Args:
      google_query (str): The query to search for.
      page (int): The page number to retrieve.
  Returns:
      A dictionary containing the search results
  """

  # using the first page
  page = page
  start = (page - 1) * 10 + 1

#   url = f"https://www.googleapis.com/customsearch/v1?key={search_engine_keys['KEY']}&cx={search_engine_keys['ID']}&q={google_query}&start={start}" + add_args
  url = "https://www.googleapis.com/customsearch/v1"
  params = {
    'key' : search_engine_keys['KEY'],
    'cx' : search_engine_keys['ID'],
    # 'dateRestrict':'y[10]',
    'fileType': '-pdf',
  }
  params.update(add_params)
#   print(url)
#   print(params)
  try:
      # Make the GET request to the Google Custom Search API
      google_response = requests.get(url, params=params)
      print(google_response.url)
      # Check if the request was successful (status code 200)
      if google_response.status_code == 200:
          # Parse the JSON response
          google_response_data = google_response.json()
          google_response_items = {}
          # get the result items
          search_items = google_response_data.get("items")
          # iterate over 10 results found
          for i, search_item in enumerate(search_items, start=1):
              try:
                  long_description = search_item["pagemap"]["metatags"][0]["og:description"]
              except KeyError:
                  long_description = "N/A"
              # get the page title
              title = search_item.get("title")
              # page snippet
              snippet = search_item.get("snippet")
              # alternatively, you can get the HTML snippet (bolded keywords)
              html_snippet = search_item.get("htmlSnippet")
              # extract the page url
              link = search_item.get("link")
              google_response_items[i] = {
                  'title': title,
                  'snippet': snippet,
                  'long_description': long_description,
                  'link': link
              }
          return google_response_items

      else:
          print(f"Error: {google_response.status_code}")
          return None
  except Exception as e:
      print(f"An error occurred: {e}")
      return None

def es_archivo_pdf(url):
    try:
        # Realizar una solicitud HEAD para obtener solo los encabezados de la respuesta
        response = requests.head(url)
        
        # Verificar si la respuesta tiene el tipo de contenido "application/pdf"
        if 'application/pdf' in response.headers.get('Content-Type', ''):
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print("Error al hacer la solicitud:", e)
        return False

def preprocess(sentence):
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    stop_words = list(set(stopwords.words('english'))) + list(set(stopwords.words('spanish')))
    word_tokens = word_tokenize(sentence.lower())
    return [word for word in word_tokens if word.isalnum() and word not in stop_words ]

def jaccard_similarity(sentence1, sentence2):
    words1 = set(preprocess(sentence1))
    words2 = set(preprocess(sentence2))
    intersection = len(words1.intersection(words2))
    try:
        if len(words1) == len(intersection):
            return 1
        else:
            union = len(words1.union(words2))
            return intersection / union if union != 0 else 0 
    except:
        union = len(words1.union(words2))
        return intersection / union if union != 0 else 0 

def check_similar (new_key, old_keys):
    for old_key in old_keys:
        similarity_score = jaccard_similarity(new_key, old_key)
        if similarity_score >= 0.7:
            return True
        else:
            continue
    return False
     
def buscar_eventos(contraseñas = None, pages=2, list_key_w= None):

    date =  dt.datetime.today().date().strftime("%Y-%m-%d")
    latest_iteration = tab1_col2.empty()
    df_events_hist = cargar_eventos_procesados_archivo(PATH_DATA + FN_EVENTS)
    df_events_busqueda = pd.DataFrame()
    df_errores_busqueda = pd.DataFrame()
    step =  1 / (10 * (pages) * len(list_key_w))
    static_3.text(f'Progreso 0 %')
    bar = tab1_col2.progress(0)
    i = 0
    # Buscar Paginas asociadas a los criterios
    for key_W in list_key_w:
        print(key_W)
        for page in range(1, pages+1):
            google_query_result = query_google_search( page, contraseñas["api_google_search"], key_W)
            if google_query_result:
                for item in google_query_result.keys():
                    df_events_hist = cargar_eventos_procesados_archivo(PATH_DATA + FN_EVENTS)
                    list_hist_links = df_events_hist['google_url'].to_list()
                    list_hist_title = df_events_hist['google_title'].to_list()
                    url = google_query_result[item]['link']
                    print("###############################################################")
                    print(url)
                    title = google_query_result[item]['title']
                    bar.progress(i+step)
                    i = i+step
                    print(f"Iteracion: {round(i*100,0)}, step:{step}")
                    if url in list_hist_links or title in list_hist_title:
                        print("Evento Ya Procesado")
                        continue
                    else:
                        static_1.markdown('**Criterio:** {}'.format(key_W['exactTerms']))
                        static_2.markdown('**Link**: {}'.format(url))
                        static_3.markdown('**Progreso:** {} %'.format(round(i*100,0)))
                        try:
                            
                            llm_result = extraer_informacion_general_gemini(url, contraseñas["api_gemini"]['KEY'])
                            # if llm_result['there_is_event'] =="Yes":
                            #     extraer_informacion_eventos_rel_gemini(url, contraseñas["api_gemini"]['KEY']):
                            if llm_result != None:
                                df_event_info = pd.DataFrame([llm_result.__dict__])
                                df_event_info ['status']  = 'OK'   
                                # df_event_info = json_to_df(llm_result)
                                df_event_info['google_title'] = google_query_result[item]['title']
                                df_event_info['google_snippet'] = google_query_result[item]['snippet']
                                df_event_info['google_long_description'] = google_query_result[item]['long_description']
                                df_event_info['google_url'] = google_query_result[item]['link']
                                df_event_info['search_criteria'] =  str(key_W)
                                df_event_info['date_processed'] =  date
                                df_event_info = limpiar_df_event(df_event_info)
                                # Filtrar y guardar eventos sin errores
                                df_event_info = df_event_info[df_event_info['status'] == "OK"]
                                df_events_hist = pd.concat([df_events_hist, df_event_info])
                                df_events_hist.to_excel(PATH_DATA + FN_EVENTS, index=False)
                                df_events_busqueda = pd.concat([df_events_busqueda, df_event_info])
                            else:
                                continue
                            
                            
                        except Exception as e:
                            print(f"Error:{e}" )
                            df_evento_error = pd.DataFrame()
                            df_evento_error ['status']  = 'ERROR'
                            df_evento_error ['error'] = e
                            df_evento_error['date_processed'] =  date
                            df_evento_error['google_url'] = google_query_result[item]['link']
                            df_errores_busqueda = pd.concat([df_errores_busqueda, df_evento_error])
                            df_errores_busqueda.to_excel(PATH_DATA + "errors_today.xlsx", index=False)
                            print(df_evento_error)
                            print(df_errores_busqueda)
                            continue

def buscar_eventos_v2(contraseñas = None, pages=2, list_key_w= None, config = {}):

    date =  dt.datetime.today().date().strftime("%Y-%m-%d")
    latest_iteration = tab1_col2.empty()
    # if 
    # df_events_hist = sf_cargar_eventos_procesados_db(contraseñas['snowflake'])
    df_events_busqueda = pd.DataFrame()
    df_errores_busqueda = pd.DataFrame()
    step =  1 / (10 * (pages) * len(list_key_w))
    static_3.text(f'Progreso 0 %')
    bar = tab1_col2.progress(0)
    i = 0
    # Buscar Paginas asociadas a los criterios
    for key_W in list_key_w:
        print(key_W)
        for page in range(1, pages+1):
            google_query_result = query_google_search( page, contraseñas["api_google_search"], key_W)
            if google_query_result:
                for item in google_query_result.keys():
                    url = google_query_result[item]['link']
                    title = google_query_result[item]['title']
                    print("###############################################################")
                    print(url)
                    
                    bar.progress(i+step)
                    i = i+step
                    
                    if (check_event_db(url, title, contraseñas, config['base_datos'])):
                            
                        print("Evento Ya Procesado")
                        continue
                    else:
                        static_1.markdown('**Criterio:** {}'.format(key_W['exactTerms']))
                        static_2.markdown('**Link**: {}'.format(url))
                        static_3.markdown('**Progreso:** {} %'.format(round(i*100,0)))
                        try:                      
                            llm_result = extraer_informacion_general_gemini(url, contraseñas["api_gemini"]['KEY'])

                            if llm_result != None:
                                df_event_info = pd.DataFrame([llm_result.__dict__])
                                df_event_info ['status']  = 'OK'   
                                # df_event_info = json_to_df(llm_result)
                                df_event_info['google_title'] = google_query_result[item]['title']
                                df_event_info['google_snippet'] = google_query_result[item]['snippet']
                                df_event_info['google_long_description'] = google_query_result[item]['long_description']
                                df_event_info['google_url'] = google_query_result[item]['link']
                                df_event_info['search_criteria'] =  str(key_W)
                                df_event_info['date_processed'] =  date
                                df_event_info = limpiar_df_event(df_event_info)
                                df_event_info['date_processed'] = pd.to_datetime(df_event_info['date_processed'])
                                # Filtrar y guardar eventos sin errores
                                df_event_info = df_event_info[df_event_info['status'] == "OK"]
                                df_events_busqueda = pd.concat([df_events_busqueda, df_event_info])
                                resultado = insert_event_db(df_event_info, contraseñas, config['base_datos'])                                    
                                if resultado == True:
                                    print("Evento Insertados Correctamente")
                                else:
                                    print("Error Insertando Evento. Error: {}".format(resultado))
                            else:
                                print(llm_result)
                        except Exception as e:
                            dict_error = {
                                'status': 'ERROR',
                                'error': str(e),
                                'date_processed' : date,
                                'google_url': url
                            }
                            print(f"Error:{e}" )
                            df_evento_error = pd.DataFrame([dict_error])
                            df_errores_busqueda = pd.concat([df_errores_busqueda, df_evento_error])
                            #df_errores_busqueda.to_excel(PATH_DATA + "errors_today.xlsx", index=False)
                            resultado = insert_event_db(df_evento_error, contraseñas, config['base_datos'])  
                                
                            if resultado == True:
                                print("Errores Insertados Correctamente")
                            else:
                                print("Error Insertando Evento. Error: {}".format(resultado))


    return df_events_busqueda


def extraer_informacion_eventos_rel_gemini(url, event, API_KEY_GEMINI):
    
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        })
    model = genai.GenerativeModel('gemini-pro')
    llm_prompt_template = """Your Task is to extract any event showed in following "Context" that can be related to the "event information". 
    "event information":{event_str}
    "context":{context_str}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=json_resp_events)

    # To extract data from WebBaseLoader
    doc_prompt = PromptTemplate.from_template("{page_content}")
    
    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str", "event_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        context = web_scrapper(url)
        if context.startswith('Not Acceptable!') or context == None:
            loader = WebBaseLoader(url)
            docs = loader.load()
            doc_prompt = PromptTemplate.from_template("{page_content}")
            context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
            
        if context != None:
            tokens_size = int(model.count_tokens(str(llm_prompt) + context).total_tokens)
            if tokens_size > 30000:
                return None
            else:
                stuff_chain = llm_prompt | llm | parser
                llm_result = stuff_chain.invoke({"context_str": context, "event_str": event} )
                return llm_result
    except Exception as e:
        print(e)
        return None

def rel_events_parser(yaml_events, df_hist_rel_events, event_key):
    df_rel_events = pd.DataFrame(columns=['event_key', 'rel_event_link', 'rel_event_key','rel_event_title', 'rel_event_year', 'rel_event_country'])

    for event in yaml_events.events:
        events_related_parsed = {}
        rel_event_key = event.title + " | " + event.country + " | " + str(event.year)
        # print(check_similar(event_key, [rel_event_key]) , check_similar(rel_event_key, df_hist_rel_events['rel_event_key']))
        if not check_similar(event_key, [rel_event_key]) and not check_similar(rel_event_key, df_hist_rel_events['rel_event_key']):
            if int(event.year) > dt.datetime.today().year -20:
                events_related_parsed['event_key'] = event_key
                events_related_parsed['rel_event_key'] = event.title + " | " + event.country + " | " + str(event.year)
                events_related_parsed['rel_event_title'] = event.title
                events_related_parsed['rel_event_country'] = event.country
                events_related_parsed['rel_event_year'] = event.year
                df_events_related_parsed = pd.DataFrame(events_related_parsed, index=[0])
                df_rel_events = pd.concat([df_rel_events, df_events_related_parsed])
    return df_rel_events

def buscar_eventos_relacionados(llm_result_event, contraseñas):

    df_rel_events = pd.DataFrame(columns=['event_key',  'rel_event_key','rel_event_title', 'rel_event_year', 'rel_event_country','rel_event_link'])
    if llm_result_event.there_is_event == "True":

        for i in range(3):

            
            if i == 0:
                add_args = {
                    'lr': 'lang_eng|lang_esp'
                }
            elif i == 1:
                add_args = {
                    'lr': 'lang_esp'
                }
            elif i == 2:
                add_args = {
                    'lr': 'lang_esp',
                    'cr': 'countryCO'
                }
            if llm_result_event.title !="" and llm_result_event.general_title !=None:
                link_or_name = llm_result_event.general_title
                search_pattern = f"related: {link_or_name} "
            else:
                link_or_name = llm_result_event.title
                search_pattern = f"related: {link_or_name} "
            #print("Criterio Busqueda:{}".format(search_pattern))
            google_query_result = query_google_search(search_pattern, 1, contraseñas["api_google_search"],add_args)
            for url in google_query_result:
                if es_archivo_pdf(google_query_result[url]['link']):
                    continue
                else:
                    #print(google_query_result[url]['link'], search_pattern)
                    ref_event_info = "title:" + llm_result_event.title + "|" +"resume:" + llm_result_event.description + "|"+"country:" + llm_result_event.country  + "|"+"year:" + llm_result_event.year
                    ref_event_key = llm_result_event.title + " | " + llm_result_event.country + " | " + llm_result_event.year 
                    try:   
                        yaml_events_related = extraer_informacion_eventos_rel_gemini(google_query_result[url]['link'], ref_event_info , contraseñas["api_gemini"]['KEY'])
                        df_events_related_link = rel_events_parser(yaml_events_related, df_rel_events, ref_event_key)
                        df_events_related_link ['rel_event_link'] = google_query_result[url]['link']
                        df_rel_events = pd.concat([df_rel_events, df_events_related_link])
                        if len(df_rel_events) >= 5:
                            return df_rel_events
                    except Exception as e:
                        print(e)
                        continue
    return df_rel_events
    
def extraer_informacion_asistentes_gemini(url, event, API_KEY_GEMINI):
    
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        })
    model = genai.GenerativeModel('gemini-pro')
    llm_prompt_template = """Tu tarea es extraer de "context" la informacion disponible del numero de asistentes al evento {event_str} en el idioma del contexto". 
    "context":{context_str}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=eventAsist)

    # To extract data from WebBaseLoader
    doc_prompt = PromptTemplate.from_template("{page_content}")
    
    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str", "event_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        context = web_scrapper(url)
        if context.startswith('Not Acceptable!') or context == None:
            loader = WebBaseLoader(url)
            docs = loader.load()
            doc_prompt = PromptTemplate.from_template("{page_content}")
            context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
            
        if context != None:
            tokens_size = int(model.count_tokens(str(llm_prompt) + context).total_tokens)
            if tokens_size > 30000:
                return None
            else:
                stuff_chain = llm_prompt | llm | parser
                llm_result = stuff_chain.invoke({"context_str": context, "event_str": event} )
                return llm_result
    except Exception as e:
        print(e)
        return None
    
def buscar_informacion_asistentes(llm_result_event, contraseñas):
    asistants_list = []
    for i in range(3):
        
        if llm_result_event.place !=None and llm_result_event.place !="":
            location = llm_result_event.place
        elif llm_result_event.country !=None and llm_result_event.country !="":
            location = llm_result_event.country
        else:
            location = ""
        search_pattern = f"{llm_result_event.general_title} {location}"
        if i == 0:
            add_args = {
                'lr': 'lang_eng|lang_esp'
            }
        elif i == 1:
            add_args = {
                'lr': 'lang_esp'
            }
        elif i == 2:
            add_args = {
                'lr': 'lang_esp',
                'cr': 'countryCO'
            }
        google_query_result = query_google_search(search_pattern, 1, contraseñas["api_google_search"], add_args)
  
        for url in google_query_result:
            if es_archivo_pdf(google_query_result[url]['link']):
                continue
            else:
                # print(google_query_result[url]['link'], search_pattern)
                ref_event_info = "title:" + llm_result_event.title + "|" +"resume:" + llm_result_event.description + "|"+"country:" + llm_result_event.country  + "|"+"year:" + llm_result_event.year
                try:   
                    yaml_envent_asistants = extraer_informacion_asistentes_gemini(google_query_result[url]['link'], ref_event_info , contraseñas["api_gemini"]['KEY'])
                    if yaml_envent_asistants.participants not in [None, 'None', '', ' '] and not check_similar(yaml_envent_asistants.participants, asistants_list):
                        asistants_list.append(yaml_envent_asistants.participants)
                        # print(asistants_list)
                        if len(asistants_list) >=3:
                            return "|".join(asistants_list)  
                except Exception as e:
                    print(e)
                    continue
    return "|".join(asistants_list) 

def json_to_df(json_dict):

    try:
        # Intenta cargar el JSON en un DataFrame

        df = pd.DataFrame([json_dict])
        return df
    except Exception as e:
        print("Error al convertir JSON a DataFrame:", e)
        return None 
    


def main():
    

    config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)
    contraseñas = cargar_contraseñas(ACCESS_PATH)
    criterios = obtener_criterios_busqueda(config)
    with tab1:
        
            
        with st.expander("Ver Criterios de Busqueda", expanded =False):

            st.markdown("**Criterios Español** ")
            with st.container(border=True):
                tab3_criterios_esp_col1, tab3_criterios_esp_col2, tab3_criterios_esp_col3 = st.columns([3, 3, 3])

                tab3_criterios_esp_col1.markdown("***Alcance*** ")
                for criterio in config['patrones_busqueda']['Esp']['alcance']:
                    tab3_criterios_esp_col1.write(criterio)
                tab3_criterios_esp_col2.markdown("***Tipo evento*** ")
                for criterio in config['patrones_busqueda']['Esp']['tipo_evento']:
                    tab3_criterios_esp_col2.write(criterio)
                tab3_criterios_esp_col3.markdown("***Lugares*** ")
                for criterio in config['lugares_busqueda']['Esp']:
                    tab3_criterios_esp_col3.write(criterio)
            
            st.markdown("**Criterios Ingles** ")
            with st.container(border=True):
                tab3_criterios_esp_col1, tab3_criterios_esp_col2, tab3_criterios_esp_col3 = st.columns([3, 3, 3])

                tab3_criterios_esp_col1.markdown("***Alcance*** ")
                for criterio in config['patrones_busqueda']['Eng']['alcance']:
                    tab3_criterios_esp_col1.write(criterio)
                tab3_criterios_esp_col2.markdown("***Tipo evento*** ")
                for criterio in config['patrones_busqueda']['Eng']['tipo_evento']:
                    tab3_criterios_esp_col2.write(criterio)
                tab3_criterios_esp_col3.markdown("***Lugares*** ")
                for criterio in config['lugares_busqueda']['Eng']:
                    tab3_criterios_esp_col3.write(criterio)            
               
                                   
                        
        iniciar_busqueda = tab1_col1.button("Iniciar Busqueda Automatica")
        if iniciar_busqueda:
            static_0.write(f"⏳ Buscando Informacion de eventos!!") 
            df_events = buscar_eventos_v2(contraseñas, pages=config['paginas'], list_key_w= criterios, config= config)
            static_0.write(f"✔️ Hemos finalizado la busqueda de eventos ")   
            with st.expander("Ver Resultados Encontrados:"):
                with st.container():
                    st.write("***Eventos encontrados:***")
                    st.dataframe(df_events, use_container_width=True, hide_index  = True)
            
    with tab2:

        df_events_hist = cargar_eventos_procesados_db(contraseñas, config['base_datos'])
        df_events_hist['date_processed'] = pd.to_datetime(df_events_hist['date_processed'])
        df_events_hist['there_is_event'] = df_events_hist['date_processed'].astype(bool)
        df_events_hist_filter = df_events_hist[(df_events_hist['status'] == "OK") &
                                                (df_events_hist['there_is_event'] == True) &
                                                (df_events_hist['country'] == "Colombia") &
                                                ((df_events_hist['year_parsed'] >= dt.datetime.today().year-10) | (df_events_hist['year_parsed'] == None))]
        cols = ['title', 'google_url', 'country', 'city', 'year', 'date', 'description', 'date_processed']
        df_events_hist_filter = df_events_hist_filter[cols]
        cols_name = ['Event title', 'Event URL', 'Event Country', 'Event City', 'Event Year', 'Event Date', 'Event Description', 'Processing Date']
        df_events_hist_filter.columns = cols_name
        column_config={"Event URL": st.column_config.LinkColumn("Event URL")}
        st.dataframe(filtrar_df(df_events_hist_filter), use_container_width=True, hide_index  = True, column_config= column_config)

if __name__ == "__main__":
    main()