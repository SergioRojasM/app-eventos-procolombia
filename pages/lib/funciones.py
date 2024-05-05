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

class event(BaseModel):

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
    
class event_v2(BaseModel):
    there_is_event: str = Field(..., description="Defines if any asociative event is mentioned. If so answer 'Yes', if not answer 'No'")
    title: Optional[str] = Field(description="The name of the event, dont use Acronyms, dont use colon punctuation")
    general_title: Optional[str] = Field(description="The name of the event, dont use Acronyms, don't use colon punctuation, don't specify the version of the event")
    date: Optional[str] = None 
    year: Optional[str] = Field(description="The year of the event, only one year. if not sure use None")
    description: Optional[str] = Field(description="Summary of the event")
    country: Optional[str] = Field(description="The location of the event, if not sure use None")
    city: Optional[str] = Field(description="The city of the event, if not sure use None")
    place: Optional[str] = Field(description="The name of the place where the event takes place, if not sure use None")
    key_words: Optional[str] = Field(description="Only five key words of thats describe de event, separated by comma, if not sure use None")
    asistants: Optional[str] = Field(description="Information about number of asistants to the event, if not sure use None")
    event_type: Optional[str] = Field(..., description="describes the event type, if no event use None ", 
                                      enum= ['Medical Sciences','Science','Social Sciences','Management','Education','Law','Economics','Technology','Industry',
                                            'Culture & Ideas','Arts','Commerce','Mathematics & Statistics','Safety & Security','Sports & Leisure','Ecology & Environment',
                                            'Transport & Communication','Historical Sciences','Library & Information', 'Other', 'None'])

class event_validation(BaseModel):
    there_is_event: str = Field(..., description="Defines if any asociative event is mentioned in the context, valid events are congress, symposium, conference, assembly, meeting, summit or seminary", enum = ['True', 'False'])
        
class event_type(BaseModel):
    event_type: Optional[str] = Field(None, description="describes the event type ", 
                                      enum= ['Medical Sciences','Science','Social Sciences','Management','Education','Law','Economics','Technology','Industry',
                                            'Culture & Ideas','Arts','Commerce','Mathematics & Statistics','Safety & Security','Sports & Leisure','Ecology & Environment',
                                            'Transport & Communication','Historical Sciences','Library & Information', 'Other'])

class event_already_evaluation(BaseModel):
    are_same_event: Optional[conint(ge=0, le=100)] = Field(None, description="describes the degree of similarity between the two events through a value between 0 and 100, where 100 represents that they are exactly the same ")
  
class event_type_enum(Enum):
    medical_sciences = 'Medical Sciences',
    science = 'Science',
    social_sciences = 'Social Sciences'
    
class event_v3(BaseModel):
    there_is_event: str = Field(..., description="Defines if any asociative event is mentioned in the context, valid events are congress, symposium, conference, assembly, meeting, summit or seminary.", enum = ['True', 'False'])
    event_type: Optional[str] = Field(None, description="describes the event type including congress, symposium, conference, assembly, meeting, summit or seminary",
                                      enum= ['Congress','Symposium','Conference','assembly','meeting','summit','seminary', 'Other'])
    title: Optional[str] = Field(None, description="The name of the event, dont use Acronyms, dont use colon punctuation")
    general_title: Optional[str] = Field(None,description="The name of the event, dont use Acronyms, don't use colon punctuation, don't specify the version of the event")
    date: Optional[str] = None 
    year: Optional[str] = Field(None,description="The year of the event, only one year. if not sure use None")
    description: Optional[str] = Field(None,description="Summary of the event with details of the event")
    country: Optional[str] = Field(None,description="The location of the event, if not sure use None")
    city: Optional[str] = Field(None,description="The city of the event, if not sure use None")
    place: Optional[str] = Field(None,description="The name of the place where the event takes place, if not sure use None")
    key_words: Optional[str] = Field(None,description="Only five key words of thats describe de event, separated by comma, if not sure use None")
    asistants: Optional[str] = Field(None,description="Information about number of asistants to the event, if not sure use None")
    event_category: Optional[str] = Field(None, description="describes the category of the event", 
                                      enum= ['Medical Sciences','Science','Social Sciences','Management','Education','Law','Economics','Technology','Industry',
                                            'Culture & Ideas','Arts','Commerce','Mathematics & Statistics','Safety & Security','Sports & Leisure','Ecology & Environment',
                                            'Transport & Communication','Historical Sciences','Library & Information', 'Other'])
    
class envent_list(BaseModel):
    events: List[event_v3] = Field(..., description="The Event details")  

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
    #with open(nombre_archivo, 'r') as f:
    #    contraseñas = toml.load(f)
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

  # using the first page
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

def web_scrapper(url):
    """

    """
    text = None
    try:
        lang_request = TextRequestsWrapper()
        lang_request.get(url)
        result = lang_request.get(url)

        bs_result = BeautifulSoup(result, features="html.parser")
        # Calculating result
        text = bs_result.get_text()
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        print('LangChain')
    except  Exception as e:
        print(f'Error LangChain: {e}')
        text = None
    
    if text == None:
        try:
            # Sending HTTP GET request to the URL
            response = requests.get(url, verify=False)
            response.raise_for_status()  # Raise an error for bad status codes

            # Parsing the HTML content of the webpage
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extracting text content from the parsed HTML
            text = soup.get_text(separator=' ', strip=True)

        except  Exception as e:
            print(f'Error request: {e}')
            text = None
    
    if text == None:
        print('return:', text)
        return None
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


def extraer_informacion_general_gemini(url, API_KEY_GEMINI):
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        })
    model = genai.GenerativeModel('gemini-pro')
    llm_prompt_template = """Your Task is to extract any event showed in following "Context", in the answer the usage of colon punctuation marks or two points punctuation marks is prohibited, use comma instead. 
    "context":{context_str}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=event)
    # parser = JsonOutputParser(pydantic_object=event)

    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    

    context = web_scrapper(url)
    if  context == None:
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
        context = context.replace(":", ",").replace(";", ",")
    elif context.startswith('Not Acceptable!'):
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
        context = context.replace(":", ",").replace(";", ",")

        
    if context != None:
        tokens_size = int(model.count_tokens(str(llm_prompt) + context).total_tokens)
        if tokens_size > 30000:
            return None
        else:
            stuff_chain = llm_prompt | llm | parser
            llm_result = stuff_chain.invoke({"context_str": context} )

            return llm_result

def extraer_info_evento_gemini(event_context, API_KEY_GEMINI):
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
    llm_prompt_template = """Your Task is to extract  information of maximum five events, only are allowed events like congress, symposium, conference, assembly, meeting, summit or seminary in the following "Context", the answer must be in english and in the answer the usage of colon ":" and ";" punctuation marks or two points punctuation marks is prohibited. 
    "context":{context_str}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=envent_list)
    # parser = JsonOutputParser(pydantic_object=event)

    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    stuff_chain = llm_prompt | llm | parser
    llm_result = stuff_chain.invoke({"context_str": event_context} )
    return llm_result

def extraer_informacion_general_gemini_v3(url, API_KEY_GEMINI):
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
    llm_prompt_template = """Your task to validate if there is any asociative event like congres, symposium, conference, assembly, meeting, summit, seminary in the following "Context". 
    "context":{context_str}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=event_validation)
    # parser = JsonOutputParser(pydantic_object=event)

    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    

    context = web_scrapper(url)
    if  context == None:
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
        context = context.replace(":", ",").replace(";", ",")
    elif context.startswith('Not Acceptable!'):
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
        context = context.replace(":", ",").replace(";", ",")
    context = context.replace(":", ",").replace(";", ",")
    
    llm_event_val_result = None
    llm_info_result = None
    context_words = 0    
    if context != None:
        context_words = len(context.split())
        tokens_size = int(model.count_tokens(str(llm_prompt) + context).total_tokens)
        if tokens_size > 30000:
            print(f"Error General: Limite de Tamaño de Token excedido {tokens_size}")
        else:

            stuff_chain = llm_prompt | llm | parser
            llm_event_val_result = stuff_chain.invoke({"context_str": context} )
            if llm_event_val_result.there_is_event == "True" or llm_event_val_result == True:
                llm_info_result = extraer_info_evento_gemini(context, API_KEY_GEMINI)

                
    return llm_event_val_result, llm_info_result, tokens_size, context_words

def extraer_tipo_evento_gemini(event_context):

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        })
    model = genai.GenerativeModel('gemini-pro')
    llm_prompt_template = """Your Task is to classificate the event in one category, if not sure use "other". 
    "context":{context_str}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=event_type)
    # parser = JsonOutputParser(pydantic_object=event)

    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    if event_context != None:
        tokens_size = int(model.count_tokens(str(llm_prompt) + event_context).total_tokens)
        print(f"Tamaño Token : {tokens_size}")
        if tokens_size > 30000:
            return None
        else:
            stuff_chain = llm_prompt | llm | parser
            llm_result = stuff_chain.invoke({"context_str": event_context} )

            return llm_result

def extraer_informacion_general_gemini_v2(url, API_KEY_GEMINI):
    print(url)
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
    llm_prompt_template = """Your Task is to extract information about any asociative events, like congres, symposium, conference, assembly, meeting, summit, seminary in the following "Context", the answer must be in english and in the answer the usage of colon punctuation marks or two points punctuation marks is prohibited. 
    "context":{context_str}
    \n{format_instructions}\n
    """
    parser = YamlOutputParser(pydantic_object=envent_list)
    # parser = JsonOutputParser(pydantic_object=event)

    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    

    context = web_scrapper(url)
    if  context == None:
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
        context = context.replace(":", ",").replace(";", ",")
    elif context.startswith('Not Acceptable!'):
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
        context = context.replace(":", ",").replace(";", ",")

        
    if context != None:
        tokens_size = int(model.count_tokens(str(llm_prompt) + context).total_tokens)
        if tokens_size > 30000:
            return None
        else:
            try:
                #stuff_chain = llm_prompt | llm | parser
                stuff_chain = llm_prompt | llm
                llm_result_list = stuff_chain.invoke({"context_str": context} )
                llm_result_list_parsed = parser.parse(llm_result_list.content)
                #for llm_re
                #if llm_result.there_is_event == 'True' or llm_result.there_is_event == True:
                #    llm_tipo_evento = extraer_tipo_evento_gemini(context)
                #else:
                #    llm_tipo_evento = None
                return llm_result_list_parsed
            except OutputParserException as e:
                print(f"Error Parsing")
                #print(llm_result_list.content.replace(":", ",").replace(";", ","))
                #new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
                #llm_result_list_parsed = new_parser.parse(llm_result_list.content.replace(":", ",").replace(";", ","))
                llm_result_list_parsed = output_parser_fix(str(llm_result_list.content))
                return llm_result_list_parsed
            except Exception as e:
                print(f"Error General: {e}")
                     
                return None

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

def get_embedding_gemini(text, API_KEY_GEMINI ):
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI

    llm = ChatGoogleGenerativeAI(model= "gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="semantic_similarity")

    vector = embeddings.embed_query(text)
    return vector

def comparar_eventos_gemini(event1, event2, API_KEY_GEMINI):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        })
    model = genai.GenerativeModel('gemini-pro')
    llm_prompt_template = """Your Task is ti evaluate if the two events given are the same event, your answer must be a value between 0 and 100, where 100 is the higest value and means that event1 and event2 are the same. 
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
            url = enlace.get('href')
            if url and not url.startswith('#'):
                url_absoluta = urljoin(sitio_web, url)
                if url_absoluta not in urls:
                    urls.append(url_absoluta)
        return urls
    else:
        print("Error al obtener la página:", respuesta.status_code)
        return []