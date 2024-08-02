import pandas as pd
import os
import streamlit as st
import os, toml, requests
import requests
import datetime as dt
import pandas as pd
import numpy as np
import nltk, json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Union
from langchain.utilities import TextRequestsWrapper
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser, YamlOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field, conint
# from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema.prompt_template import format_document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    GoogleGenerativeAIEmbeddings
)
import google.generativeai as genai
import unicodedata
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from pages.lib.config import FN_KEYW_JSON, ACCESS_PATH, PATH_DATA
from pages.lib.funciones_db import mdb_get_k_nearest_results
from enum import Enum



# FUNCIONES PARA MANEJO DE CONFIGURACION

def cargar_configuracion(path_file):
    if not os.path.exists(path_file):
        configuracion = {
        "modelo": "Gemini",
        "paginas":1,
        "criterios": ["World Congress-Colombia", "Eventos-Colombia-Bogota"],
        "patrones_busqueda":{
                "Esp":{"alcance": ["Mundial", "Internacional"], "tipo_evento": ["Congreso", "Simposio"]}, 
                "Ing":{"alcance": ["World", "International"], "tipo_evento": ["Congress", "Simposium"]}
                  },
        "lugares_busqueda":{
                "Esp":["", "Universidad del Bosque"],
                "Ing":["", "Universidad del Bosque"]
                }
        }
        with open(path_file, "w") as archivo:
            json.dump(configuracion, archivo, indent=4)

    else:
        with open(path_file, 'r') as archivo:
            configuracion = json.load(archivo)

    return configuracion

def actualizar_configuracion(configuracion):
    with open(PATH_DATA + FN_KEYW_JSON, "w") as archivo:
            json.dump(configuracion, archivo, indent=4)

def cargar_contraseñas(nombre_archivo):
    return st.secrets

def obtener_criterios_busqueda(config):
    
    list_search_params = []
    periodo = 'y[10]'
    
    if config['periodo'] == 'Ultimo año':
        periodo = 'y[1]'
    elif config['periodo'] == 'Ultimo mes':
        periodo = 'm[1]'
    elif config['periodo'] == 'Ultima semana':
        periodo = 'w[1]'
    else :
        periodo = 'y[10]'
        
    if config['orden'] == 'Mas Recientes':
        orden_list = ['date']
    elif config['orden'] == 'Los dos metodos':
        orden_list = ['', 'date']
    else :
        orden_list = ['']

        
    
    for idioma in config['patrones_busqueda']:
        for alcance in config['patrones_busqueda'][idioma]['alcance']:
            for tipo_evento in config['patrones_busqueda'][idioma]['tipo_evento']:
                if idioma == "Eng":
                    for lugar in config['lugares_busqueda']['Eng']:
                        for orden in orden_list:
                            if orden == "":
                                search_params = {
                                                'q': f'+{alcance}+{tipo_evento}+Colombia+{lugar}',
                                                'lr': 'lang_esp|lang_eng',
                                                'exactTerms': f'({alcance}).({tipo_evento})',
                                                'dateRestrict': periodo
                                                }
                                list_search_params.append(search_params)
                            elif orden == 'date':
                                search_params = {
                                                'q': f'+{alcance}+{tipo_evento}+Colombia+{lugar}',
                                                'lr': 'lang_esp|lang_eng',
                                                'exactTerms': f'({alcance}).({tipo_evento})',
                                                'dateRestrict': periodo,
                                                'sort': orden
                                                }
                                list_search_params.append(search_params)
                if idioma == "Esp":
                    for lugar in config['lugares_busqueda']['Esp']:
                        for orden in orden_list:
                            if orden == "":
                                search_params = {
                                                'q': f'+{tipo_evento}+{alcance}+Colombia+{lugar}',
                                                'lr': 'lang_esp|lang_eng',
                                                'exactTerms': f'({alcance}).({tipo_evento})',
                                                'dateRestrict': periodo
                                                }
                                list_search_params.append(search_params)
                            elif orden == 'date':
                                search_params = {
                                                'q': f'+{tipo_evento}+{alcance}+Colombia+{lugar}',
                                                'lr': 'lang_esp|lang_eng',
                                                'exactTerms': f'({alcance}).({tipo_evento})',
                                                'dateRestrict': periodo,
                                                'sort': orden
                                                }
                                list_search_params.append(search_params)

    return list_search_params

def cargar_eventos_procesados_archivo(path_file):
    if not os.path.exists(path_file):
        # static_1.warning('Archivo con la base de eventos no encontrada, se creara uno en blanco en la ruta "{}".'.format(PATH_DATA), icon="⚠️")
        cols = [ 'there_is_event', 'title', 'general_title', 'date', 'year', 'description', 'country', 'city','place', 'key_words','asistants', 'status','google_title', 'google_snippet', 'google_long_description', 'google_url', 'search_criteria', 'date_processed','year_parsed']
        df_events = pd.DataFrame(columns = cols)
        df_events.to_excel(path_file, index=False)

    df_events = pd.read_excel(path_file)
    cols = {
    'there_is_event': bool,
    'title': str,
    'general_title': str,
    'date': str,
    'year': float,
    'description': str,
    'country': str,
    'city': str,
    'place': str,
    'key_words': str,
    'asistants': str,
    'status': str,
    'google_title': str,
    'google_snippet': str,
    'google_long_description': str,
    'google_url': str,
    'search_criteria': str,
    'date_processed': str,
    'year_parsed': float
    }
    df_events = df_events.astype(cols)
    return df_events

### GENERALES

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

def check_similar (new_key, old_keys):
    for old_key in old_keys:
        similarity_score = jaccard_similarity(new_key, old_key)
        if similarity_score >= 0.7:
            return True
        else:
            continue
    return False

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
     

### FUNCIONES PARA GOOGLE SEARCH

def query_google_search(page=1, search_engine_keys=None, add_params = {}):
  """
  Query the Google Custom Search API and return the results in a dictionary.

  Args:
      google_query (str): The query to search for.
      page (int): The page number to retrieve.
  Returns:
      A dictionary containing the search results
  """

  page = page
  start = (page - 1) * 10 + 1

#   url = f"https://www.googleapis.com/customsearch/v1?key={search_engine_keys['KEY']}&cx={search_engine_keys['ID']}&q={google_query}&start={start}" + add_args
  url = "https://www.googleapis.com/customsearch/v1"
  params = {
    'key' : search_engine_keys['KEY'],
    'cx' : search_engine_keys['ID'],
    'fileType': '-pdf',
  }
  params.update(add_params)

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

### FUNCIONES PARA USO DE GEMINI

def obtener_tamano_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        tamano_contenido = len(response.content)
        print(f"El tamaño del contenido en {url} es {tamano_contenido} bytes.")
    except requests.exceptions.RequestException as e:
        print(f"Error al acceder a la URL: {e}")

def web_scrapper(url):
    """

    """
    text = None
    try:
        print('Scraping: langChain')
        lang_request = TextRequestsWrapper()
        lang_request.get(url)
        result = lang_request.get(url)
        bs_result = BeautifulSoup(result, features="html.parser")
        text = bs_result.get_text()
    except  Exception as e:
        print(f'Error LangChain: {e}')
        text = None
    
    if text == None:
        try:
            print('Scraping: Requests')
            response = requests.get(url, verify=False)
            response.raise_for_status() 
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        except  Exception as e:
            print(f'Error request: {e}')
            text = None
    if text == None or text.startswith('Not Acceptable!'):
        try:
            print("Scraping: WebBaseLoader")
            loader = WebBaseLoader(url)
            docs = loader.load()
            doc_prompt = PromptTemplate.from_template("{page_content}")
            text = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
            text = text.replace(":", ",").replace(";", ",")
        except Exception as e:
            print(f'Error WebBaseLoader: {e}')
            text = None
    else:
        text = text.replace(":", ",").replace(";", ",")
    
    return text
        
def limpiar_dict_event(diccionario):
    def quitar_tildes(texto):
        if texto != None: 
            texto_normalizado = unicodedata.normalize('NFD', texto)
            texto_sin_tildes = texto_normalizado.encode('ascii', 'ignore').decode('utf-8')
            return texto_sin_tildes
        else:
            return None
    # Reemplazar "None" por NaN
    for key, value in diccionario.items():
        if value == "None":
            diccionario[key] = None
    
    # Rellenar NaN con None
    for key, value in diccionario.items():
        if value == np.nan:
            diccionario[key] = None
    
    # Reemplazar cadenas vacías por NaN
    for key, value in diccionario.items():
        if value == "":
            diccionario[key] = None
    
    diccionario['city'] = quitar_tildes(diccionario['city'])
    
    try:
        diccionario['year_parsed'] = float(diccionario['year'])
    except:
        diccionario['year_parsed'] = None
    return diccionario

def filtrar_df(df):
    modify = st.checkbox("Crear Filtros")
    if not modify:
        return df
    
    df = df.copy()
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", ['Event Country', 'Event City', 'Event Year', 'Processing Date'])
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            if column == 'Processing Date':
                df[column] = pd.to_datetime(df[column])
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
    return df

def get_embedding_gemini(text, API_KEY_GEMINI ):
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI

    llm = ChatGoogleGenerativeAI(model= "gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="semantic_similarity")

    vector = embeddings.embed_query(text)
    return vector

def comparar_eventos_gemini(event1, event2, API_KEY_GEMINI):
    
    
    class event_already_evaluation(BaseModel):
        are_same_event: Optional[conint(ge=0, le=100)] = Field(None, description="describes the degree of similarity between the two events through a value between 0 and 100, where 100 represents that they are exactly the same ")

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        })
    model = genai.GenerativeModel('gemini-pro')
    llm_prompt_template = """Your Task is to evaluate if the two events given are the same event, your answer must be a value between 0 and 100, where 100 is the higest value and means that event1 and event2 are the same. 
    "event1":{event1}
    "event2":{event2}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=event_already_evaluation)
    # parser = JsonOutputParser(pydantic_object=event)

    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["event1", "event2"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    
    stuff_chain = llm_prompt | llm | parser
    llm_result = stuff_chain.invoke({"event1": event1, "event2": event2} )

    return llm_result

def get_embedding_gemini(text, API_KEY_GEMINI ):
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="semantic_similarity")

    vector = embeddings.embed_query(text)
    return vector

def check_event_embedding_gemini(event_in, contraseñas):
    # query = 'Congreso Internacional de Odontología'
    query = f"{event_in.title}, {event_in.description},  {event_in.date}, {event_in.year}, {event_in.country}, {event_in.city}"
    embedding = get_embedding_gemini(query, contraseñas["api_gemini"]['KEY'])
    k_events  = mdb_get_k_nearest_results(embedding, 3, 'fct_eventos_turismo', contraseñas["mongo_db"])
    flag_event = False

    if k_events:
        for event_db in k_events:
            event_db_text = f"{event_db['title']}, {event_db['description']},  {event_db['date']}, {event_db['year']}, {event_db['country']}, {event_db['city']}"
            llm_result = comparar_eventos_gemini(event_in, event_db_text, contraseñas["api_gemini"]['KEY'])
            # print("Nuevo Evento: {}".format(event_in.title))
            # print("Evento DB: {}".format(event_db['title']))
            
            if llm_result.are_same_event>90:
                flag_event = True
            # print(f"Comparacion: {llm_result}, resultado:{flag_event}")
                
    return flag_event

def buscar_urls_pagina(sitio_web):
    respuesta = requests.get(sitio_web)
    if respuesta.status_code == 200:
        soup = BeautifulSoup(respuesta.text, 'html.parser')
        enlaces = soup.find_all('a')
        urls = []
        for enlace in enlaces:
            try:
                url = enlace.get('href')
                if url and not url.startswith('#'):
                    url_absoluta = urljoin(sitio_web, url)
                    if url_absoluta not in urls:
                        urls.append(url_absoluta)
            except:
                continue
        return urls
    else:
        print("Error al obtener la página:", respuesta.status_code)
        return []