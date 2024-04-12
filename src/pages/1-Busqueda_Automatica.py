
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

from pages.lib.funciones import extraer_informacion_general_gemini, filtrar_df


# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'

FN_KEYW = 'db_eventos_keyw.xlsx'
FN_EVENTS = 'events_data.xlsx'
FN_EVENTS_TODAY = 'events_data_today.xlsx'
FN_EVENTS_FILTER = 'events_data_filter.xlsx'
FN_KEYW_JSON = 'app_config.json'
ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"
#
MODELS_DICT = {'Gemini':0, 'GROG-LLAMA2':1}

pd.set_option('future.no_silent_downcasting', True)
# Configuracion de la pagina
st.set_page_config(page_title="Busqueda Automatica", page_icon=":rocket:",layout="wide")
st.image(PATH_IMG + "header_verde.jpg")
st.subheader("Busqueda de Eventos de Turismo")
menu()
tab1, tab2, tab3 = st.tabs(["Busqueda Automatica", "Resultados", "Configuración"])
tab1_col1, tab1_col2 = tab1.columns([2, 5])
static_0 = tab1_col2.empty()
static_1 = tab1_col2.empty()
static_2 = tab1_col2.empty()
static_3 = tab1_col2.empty()
# Define your desired data structure.

class event(BaseModel):
    resume: str = Field(description="The resume of the context in few words")
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

@st.cache_resource
def cargar_contraseñas(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        contraseñas = toml.load(f)
    return contraseñas

def cargar_llm(GEMINI_API):

    os.environ["GOOGLE_API_KEY"] = GEMINI_API
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    return llm

def cargar_configuracion():
    
    if not os.path.exists(PATH_DATA + FN_KEYW_JSON):
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
        with open(PATH_DATA + FN_KEYW_JSON, "w") as archivo:
            json.dump(configuracion, archivo, indent=4)

    else:
        with open(PATH_DATA + FN_KEYW_JSON, 'r') as archivo:
            configuracion = json.load(archivo)
        # lista_criterios = []
        # for criterio in configuracion['criterios']:
        #     palabras = []
        #     palabras = ['"{}"'.format(cadena) for cadena in criterio.split("-")]
        #     lista_criterios.append(' + '.join(palabras))
        # configuracion['criterios'] = lista_criterios
    return configuracion

def obtener_criterios_busqueda(config):
    list_search_params = []
    for idioma in config['patrones_busqueda']:
        for alcance in config['patrones_busqueda'][idioma]['alcance']:
            for tipo_evento in config['patrones_busqueda'][idioma]['tipo_evento']:
                if idioma == "Eng":
                    for lugar in config['lugares_busqueda']['Eng']:
                        search_params = {
                                        'q': f'+{tipo_evento}+Colombia+{lugar}',
                                        'lr': 'lang_esp|lang_eng',
                                        'exactTerms': f'({alcance}).({tipo_evento})'
                                        }
                        # search_params = {
                        #                 'tipo_evento':tipo_evento,
                        #                 'alcance':alcance,
                        #                 'lugar': lugar,
                        #                 'lang':'lang_eng'
                        #                 }
                        list_search_params.append(search_params)
                if idioma == "Esp":
                    for lugar in config['lugares_busqueda']['Esp']:
                        search_params = {
                                        'q': f'+{tipo_evento}+colombia+{lugar}',
                                        'lr': 'lang_esp|lang_eng',
                                        'exactTerms': f'({tipo_evento}).({alcance})'
                                        }
                        # search_params = {
                        #                 'tipo_evento':tipo_evento,
                        #                 'alcance':alcance,
                        #                 'lugar': lugar,
                        #                 'lang':'lang_esp'
                        #                 }
                        list_search_params.append(search_params)
    return list_search_params

def actualizar_configuracion(configuracion):
    with open(PATH_DATA + FN_KEYW_JSON, "w") as archivo:
            json.dump(configuracion, archivo, indent=4)

def cargar_eventos_procesados_archivo():
    if not os.path.exists(PATH_DATA + FN_EVENTS):
        # static_1.warning('Archivo con la base de eventos no encontrada, se creara uno en blanco en la ruta "{}".'.format(PATH_DATA), icon="⚠️")
        cols = ['resume', 'there_is_event', 'title', 'general_title', 'date', 'year', 'description', 'country', 'city','place', 'key_words','asistants', 'status','google_title', 'google_snippet', 'google_long_description', 'google_url', 'search_criteria', 'date_processed','year_parsed']
        df_events = pd.DataFrame(columns = cols)
        df_events.to_excel(PATH_DATA + FN_EVENTS, index=False)

    df_events = pd.read_excel(PATH_DATA + FN_EVENTS)
    cols = {
    'resume': str,
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
    'dateRestrict':'y[10]',
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

# def web_scrapper(url):
#   """

#   """
#   # Initializing variable
#   lang_request = TextRequestsWrapper()
#   try:
#     lang_request.get(url)
#   except:
#     print('ERR', 'Error scrappiing the url')
#     return None
#   # Initializing variable
#   result = lang_request.get(url)
#   # Initializing variable
#   bs_result = BeautifulSoup(result, features="html.parser")
#   # Calculating result
#   text = bs_result.get_text()
#   text = text.replace("\n", " ")
#   text = text.replace("\t", " ")
#   return text

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

# def extraer_informacion_general_gemini(url, API_KEY_GEMINI):
#     os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-pro", 
#         safety_settings={
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         })
#     model = genai.GenerativeModel('gemini-pro')
#     llm_prompt_template = """Your Task is to extract any event showed in following "Context", in the answer the usage of colon punctuation marks or two points punctuation marks is prohibited, use comma instead. 
#     "context":{context_str}
#     \n{format_instructions}\n
#     """
#     parser = YamlOutputParser(pydantic_object=event)
#     # parser = JsonOutputParser(pydantic_object=event)

#     # Realizar el query a Gemini
#     llm_prompt = PromptTemplate.from_template(llm_prompt_template)

#     llm_prompt = PromptTemplate(
#         template=llm_prompt_template,
#         input_variables=["context_str"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )
    
#     try:
#         context = web_scrapper(url)
#         if context.startswith('Not Acceptable!') or context == None:
#             loader = WebBaseLoader(url)
#             docs = loader.load()
#             doc_prompt = PromptTemplate.from_template("{page_content}")
#             context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
            
#         if context != None:
#             tokens_size = int(model.count_tokens(str(llm_prompt) + context).total_tokens)
#             if tokens_size > 30000:
#                 return None
#             else:
#                 stuff_chain = llm_prompt | llm | parser
#                 llm_result = stuff_chain.invoke({"context_str": context, "event_str": event} )

#                 return llm_result
#     except Exception as e:
#         print(e)
#         return None

def limpiar_df_event(df):
    
    def cast_year(val):
        try:
            val = float(val)
        except:
            val = np.nan
    
    df = df.replace("None", np.nan)
    df = df.fillna(np.nan)
    df = df.replace("", np.nan)
    df['there_is_event'] = df['there_is_event'].apply(lambda x: x if x in [True, False, "True", "False"] else False)
    # df['there_is_event'] = df['there_is_event'].replace(to_replace={"True": True, "False": False}, value=False)
    # df['there_is_event'] = df['there_is_event'].replace(to_replace={True: True, False: False}, value=False)
    df['year_parsed'] = df['year']
    # df['year_parsed'] = df['there_is_event'].apply(cast_year)
    df['year_parsed'] = pd.to_numeric(df['year'], errors='coerce')
    cols = {
    'resume': str,
    'there_is_event': bool,
    'title': str,
    'general_title': str,
    'date': str,
    'year': str,
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
    'year_parsed':float
    }
    df = df.astype(cols)
    return df
         

def buscar_eventos(contraseñas = None, pages=2, list_key_w= None):

    date =  dt.datetime.today().date().strftime("%Y-%m-%d")
    latest_iteration = tab1_col2.empty()
    df_events_hist = cargar_eventos_procesados_archivo()
    df_events_busqueda = pd.DataFrame()
    df_errores_busqueda = pd.DataFrame()
    step =  int(100 / (10 * (pages) * len(list_key_w) ))
    static_3.text(f'Progreso 0 %')
    bar = tab1_col2.progress(0)
    i = 0
    # Buscar Paginas asociadas a los criterios
    for key_W in list_key_w:
        


        for page in range(1, pages+1):
            google_query_result = query_google_search( page, contraseñas["api_google_search"], key_W)
            if google_query_result:
                for item in google_query_result.keys():
                    list_hist_links = df_events_hist['google_url'].to_list()
                    list_hist_title = df_events_hist['google_title'].to_list()
                    url = google_query_result[item]['link']
                    print(url)
                    title = google_query_result[item]['title']
                    bar.progress(i+step)
                    i = i+step
                    if url in list_hist_links or title in list_hist_title:
                        print("Evento Ya Procesado")
                        continue
                    else:
                        static_1.markdown('**Criterio:** {}'.format(key_W['exactTerms']))
                        static_2.markdown('**Link**: {}'.format(url))
                        static_3.markdown('**Progreso:** {} %'.format(i))
                        try:
                            llm_result = extraer_informacion_general_gemini(url, contraseñas["api_gemini"]['KEY'])
                            # if llm_result['there_is_event'] =="Yes":
                            #     extraer_informacion_eventos_rel_gemini(url, contraseñas["api_gemini"]['KEY']):
                        except Exception as e:
                            llm_result = None
                            
                            print(e)
                        finally:
                            if llm_result == None:
                                df_evento_error = pd.DataFrame()
                                df_evento_error ['status']  = 'ERROR'
                                df_evento_error ['eror'] = e
                                df_evento_error['date_processed'] =  date
                                df_evento_error['google_url'] = google_query_result[item]['link']
                                df_errores_busqueda = pd.concat([df_errores_busqueda, df_evento_error])
                                df_errores_busqueda.to_excel(PATH_DATA + "errors_today.xlsx", index=False)
                                continue
                            else:
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
                                df_events_busqueda = pd.concat([df_events_busqueda, df_event_info])
                                df_events_busqueda.to_excel(PATH_DATA + "events_data_today.xlsx", index=False)
                                df_events_hist = pd.concat([df_events_hist, df_event_info])
                                df_events_hist.to_excel(PATH_DATA + FN_EVENTS, index=False)
                        
    df_events_hist = pd.read_excel(PATH_DATA + FN_EVENTS)
    df_events_hist_filter = df_events_hist[(df_events_hist['status'] == "OK") &
                                (df_events_hist['there_is_event'] == True) &
                                (df_events_hist['country'] == "Colombia") &
                                ((df_events_hist['year'] >= dt.datetime.today().year-10) | (df_events_hist['year'] == None))]
    df_events_hist_filter.to_excel(PATH_DATA + FN_EVENTS_FILTER, index=False)
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
                    ref_event_info = "title:" + llm_result_event.title + "|" +"resume:" + llm_result_event.resume + "|"+"country:" + llm_result_event.country  + "|"+"year:" + llm_result_event.year
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
                ref_event_info = "title:" + llm_result_event.title + "|" +"resume:" + llm_result_event.resume + "|"+"country:" + llm_result_event.country  + "|"+"year:" + llm_result_event.year
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
    

    config = cargar_configuracion()
    contraseñas = cargar_contraseñas(ACCESS_PATH)
    criterios = obtener_criterios_busqueda(config)
    with tab1:
        
            # with st.expander("Ver Criterios de Busqueda"):
            #     with st.container():
            #         st.write("***Critrios:***")
                    
            #         for i, key in enumerate([criterios]):
            #             st.write(f"{i+1} - {str(key)}")
            
        with st.expander("Ver Criterios de Busqueda", expanded =False):
            config = cargar_configuracion()
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
            df_events = buscar_eventos(contraseñas, pages=config['paginas'], list_key_w= criterios)
            static_0.write(f"✔️ Hemos finalizado la busqueda de eventos ")   
            with st.expander("Ver Resultados Encontrados:"):
                with st.container():
                    st.write("***Eventos encontrados:***")
                    df_events_hist = pd.read_excel(PATH_DATA + FN_EVENTS_TODAY)
                    df_events_hist_filter = df_events_hist[(df_events_hist['status'] == "OK") &
                                                        (df_events_hist['there_is_event'] == True) &
                                                        (df_events_hist['country'] == "Colombia") &
                                                        ((df_events_hist['year'] >= dt.datetime.today().year-10) | (df_events_hist['year'] == None))]
                    st.dataframe(df_events_hist_filter, use_container_width=True, hide_index  = True)
            
    with tab2:

        df_events_hist = cargar_eventos_procesados_archivo()
        df_events_hist_filter = df_events_hist[(df_events_hist['status'] == "OK") &
                                                (df_events_hist['there_is_event'] == True) &
                                                (df_events_hist['country'] == "Colombia") &
                                                ((df_events_hist['year'] >= dt.datetime.today().year-10) | (df_events_hist['year'] == None))]
        cols = ['title', 'google_url', 'country', 'city', 'year', 'date', 'description', 'date_processed']
        df_events_hist_filter = df_events_hist_filter[cols]
        cols_name = ['Event title', 'Event URL', 'Event Country', 'Event City', 'Event Year', 'Event Date', 'Event Description', 'Processing Date']
        df_events_hist_filter.columns = cols_name
        column_config={"Event URL": st.column_config.LinkColumn("Event URL")}
        st.dataframe(filtrar_df(df_events_hist_filter), use_container_width=True, hide_index  = True, column_config= column_config)
        
        
    with tab3:
        st.header("Configuracion por defecto")

        st.markdown("***Modelo LLM*** ")
        config['modelo'] = st.radio(
                                "Seleccione un modelo 👉",
                                key="model",
                                options=["Gemini", "GROG-LLAMA2"],
                                index= MODELS_DICT[config['modelo']],
                                horizontal = True
                            )
        
        st.markdown("***Numero de Paginas a buscar en google por Criterio*** ")
        config['paginas'] = st.radio(
                        "Seleccione numero de paginas 👉",
                        key="pages",
                        options=[1, 2, 3, 4, 5],
                        index= config['paginas']-1,
                        horizontal = True
                    )
        
        st.markdown("***Criterios de Busqueda*** ")
        with st.expander("Ver Criterios de Busqueda", expanded =False):
            config = cargar_configuracion()
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
                
        st.markdown("***Agregar Criterios de Busqueda*** ")
        info_1= st.empty()
        info_2= st.empty()
        info_3= st.empty()
        col1, col2, col3= st.columns([2, 4,4])
        add_cri_chk_b = col1.checkbox('Agregar', key="add_cri")
        if add_cri_chk_b:

            to_add_alcance = col2.text_input("Ingrese un nuevo alcance de eventos:", key="add_key_alcance")
            to_add_tipo = col2.text_input("Ingrese un nuevo tipo de eventos:", key="add_key_tipo")
            to_add_lugar = col2.text_input("Ingrese un nuevo lugar de eventos:", key="add_key_lugar")
            
            idioma_radio_add = col3.radio("Seleccione el idioma 👉", ['Esp', 'Eng'], horizontal =False,  key="radio_idioma_add")
            if col3.button("Configuracion configuracion", key="add_key"):
                config_modificada = False
                if idioma_radio_add == 'Esp':
                    if to_add_alcance:
                        if to_add_alcance in config['patrones_busqueda']['Esp']['alcance']:
                            info_1.warning(f'"**{to_add_alcance}**" ya esta configurado como alcance!!', icon="⚠️")
                        else:
                            config['patrones_busqueda']['Esp']['alcance'].append(to_add_alcance)
                            config_modificada = True
                            actualizar_configuracion(config)
                            info_1.markdown(f'"✔️ Alcance **{to_add_alcance}**" Se adiciono a la configuracion!!!')
                             
                    if to_add_tipo:
                        if to_add_tipo in config['patrones_busqueda']['Esp']['tipo_evento']:
                            info_2.warning(f'"**{to_add_tipo}**" ya esta configurado como alcance!!', icon="⚠️")
                        else:
                            config['patrones_busqueda']['Esp']['tipo_evento'].append(to_add_tipo)
                            config_modificada = True
                            actualizar_configuracion(config)
                            info_2.markdown(f'"✔️ Tipo de evento **{to_add_tipo}**" Se adiciono a la configuracion!!!')
                        
                    if to_add_lugar:
                        if to_add_lugar in config['lugares_busqueda']['Esp']:
                            info_3.warning(f'"**{to_add_lugar}**" ya esta configurado como alcance!!', icon="⚠️")
                        else:
                            config['lugares_busqueda']['Esp'].append(to_add_lugar)
                            config_modificada = True
                            actualizar_configuracion(config)
                            info_3.markdown(f'"✔️ Tipo de evento **{to_add_lugar}**" Se adiciono a la configuracion!!!')
                          
                    
                elif idioma_radio_add == 'Eng':
                    if to_add_alcance:
                        if to_add_alcance in config['patrones_busqueda']['Eng']['alcance']:
                            info_1.warning(f'"**{to_add_alcance}**" ya esta configurado como alcance!!', icon="⚠️")
                        else:
                            config['patrones_busqueda']['Eng']['alcance'].append(to_add_alcance)
                            config_modificada = True
                            actualizar_configuracion(config)
                            info_1.markdown(f'"✔️ Alcance **{to_add_alcance}**" Se adiciono a la configuracion!!!')
                             
                    if to_add_tipo:
                        if to_add_tipo in config['patrones_busqueda']['Eng']['tipo_evento']:
                            info_2.warning(f'"**{to_add_tipo}**" ya esta configurado como alcance!!', icon="⚠️")
                        else:
                            config['patrones_busqueda']['Eng']['tipo_evento'].append(to_add_tipo)
                            config_modificada = True
                            actualizar_configuracion(config)
                            info_2.markdown(f'"✔️ Tipo de evento **{to_add_tipo}**" Se adiciono a la configuracion!!!')
                        
                    if to_add_lugar:
                        if to_add_lugar in config['lugares_busqueda']['Eng']:
                            info_3.warning(f'"**{to_add_lugar}**" ya esta configurado como alcance!!', icon="⚠️")
                        else:
                            config['lugares_busqueda']['Eng'].append(to_add_lugar)
                            config_modificada = True
                            actualizar_configuracion(config)
                            info_3.markdown(f'"✔️ Tipo de evento **{to_add_lugar}**" Se adiciono a la configuracion!!!')
                
            
        st.markdown("***Eliminar Criterios de Busqueda*** ")
        info_4 = st.empty() 
        info_5 = st.empty() 
        info_6 = st.empty()       
        col1, col2= st.columns([2, 6])
        rm_cri_chk_b = col1.checkbox('Eliminar', key="rm_cri")
        if rm_cri_chk_b:
            idioma_radio_rmv = col1.radio("Seleccione el idioma 👉", ['Esp', 'Eng'], horizontal =False, key="radio_idioma_rm")
            list_rmv_alcance = []
            list_rmv_tipo = []
            list_rmv_lugar = []
            config = cargar_configuracion()
            # if idioma_radio_rmv == "Esp":
            with st.container(border=True):
                col2_1, col2_2, col2_3 = col2.columns([3, 3, 3])

                col2_1.markdown("***Alcance*** ")
                for i, criterio in enumerate(config['patrones_busqueda'][idioma_radio_rmv]['alcance']):
                    rm_cb_alcance = col2_1.checkbox(criterio, key=f"cb_alcance{i}")
                    
                col2_2.markdown("***Tipo Evento*** ")
                for i, criterio in enumerate(config['patrones_busqueda'][idioma_radio_rmv]['tipo_evento']):
                    rm_cb_tipo= col2_2.checkbox(criterio, key=f"cb_tipo{i}")
                    
                col2_3.markdown("***Lugares*** ")
                for i, criterio in enumerate(config['lugares_busqueda'][idioma_radio_rmv]):
                    if criterio != "":
                        rm_cb_lugar= col2_3.checkbox(criterio, key=f"cb_lugar{i}")
                    
                if col1.button("Actualizar Configuracion", key="rm_key"):
                    for i, criterio in enumerate(config['patrones_busqueda'][idioma_radio_rmv]['alcance']):
                        if st.session_state[f"cb_alcance{i}"]:
                            list_rmv_alcance.append(criterio) 
                            
                    for i, criterio in enumerate(config['patrones_busqueda'][idioma_radio_rmv]['tipo_evento']):
                        if st.session_state[f"cb_tipo{i}"]:
                            list_rmv_tipo.append(criterio) 

                    for i, criterio in enumerate(config['lugares_busqueda'][idioma_radio_rmv]):
                        if criterio != "":
                            if st.session_state[f"cb_lugar{i}"]:
                                list_rmv_lugar.append(criterio) 

                    for item in list_rmv_alcance:
                        st.write(len(config['patrones_busqueda'][idioma_radio_rmv]['alcance']))
                        if len(config['patrones_busqueda'][idioma_radio_rmv]['alcance']) > 1:
                            config['patrones_busqueda'][idioma_radio_rmv]['alcance'].remove(item)
                        else:
                            info_4.warning(f'No es posible eliminar todas las opciones configuradas en alcance!!', icon="⚠️")
                            
                    for item in list_rmv_tipo:
                        if len(config['patrones_busqueda'][idioma_radio_rmv]['tipo_evento']) > 1:
                            config['patrones_busqueda'][idioma_radio_rmv]['tipo_evento'].remove(item)
                        else:
                            info_5.warning(f'No es posible eliminar todas las opciones configuradas en alcance!!', icon="⚠️")
                            
                    for item in list_rmv_lugar:
                        config['lugares_busqueda'][idioma_radio_rmv].remove(item)

                    
                    actualizar_configuracion(config)
                    # info_4.markdown(f'"✔️ Configuracion actualizada!!!')
                    
                    
                        
                    # if col1.button("Actualizar Configuracion", key="rm_key"):
                    #     for i, criterio in enumerate(config['patrones_busqueda'][idioma_radio_rmv]['alcance']):
                    #         if st.session_state[f"cb{i}"]:
                    #             list_rmv.append(criterio) 
                    #             # config['patrones_busqueda']['Esp']['alcance'].remove(criterio)
                    #     for item_rmv in list_rmv:
                    #         config['patrones_busqueda']['Esp']['alcance'].remove(item_rmv)
                    #     st.write(list_rmv)
                    #     actualizar_configuracion(config)
                                    

            
            
            
            
            # to_del_cri = col2.selectbox(
            #                 "Seleccione el criterio ",
            #                 key="df_key_w_list",
            #                 options=config['criterios'],
            #             )
        # if st.button("Actualizar Configuracion", key="rm_key"):
        #     if add_cri_chk_b:
        #         config['criterios'].append(to_add_cri)
        #     if rm_cri_chk_b:
        #         config['criterios'].remove(to_del_cri)
        #     actualizar_configuracion(config)
        #     st.write("Configuracion Actualizada!!!")
        #     st.write(config)        
        


if __name__ == "__main__":
    main()