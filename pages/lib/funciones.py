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
import google.generativeai as genai

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

#
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
                                                'q': f'+{tipo_evento}+Colombia+{lugar}',
                                                'lr': 'lang_esp|lang_eng',
                                                'exactTerms': f'({alcance}).({tipo_evento})',
                                                'dateRestrict': periodo
                                                }
                                list_search_params.append(search_params)
                            elif orden == 'date':
                                search_params = {
                                                'q': f'+{tipo_evento}+Colombia+{lugar}',
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
                                                'q': f'+{tipo_evento}+Colombia+{lugar}',
                                                'lr': 'lang_esp|lang_eng',
                                                'exactTerms': f'({alcance}).({tipo_evento})',
                                                'dateRestrict': periodo
                                                }
                                list_search_params.append(search_params)
                            elif orden == 'date':
                                search_params = {
                                                'q': f'+{tipo_evento}+Colombia+{lugar}',
                                                'lr': 'lang_esp|lang_eng',
                                                'exactTerms': f'({alcance}).({tipo_evento})',
                                                'dateRestrict': periodo,
                                                'sort': orden
                                                }
                                list_search_params.append(search_params)

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
        

def limpiar_df_event(df):
    # Reemplazar "None" por NaN
    df = df.replace("None", np.nan)
    
    # Rellenar NaN con None
    df = df.fillna(value=np.nan)
    
    # Reemplazar cadenas vacías por NaN
    df = df.replace("", np.nan)
    
    # Convertir NaN de nuevo a None
    df = df.where(pd.notnull(df), None)
    
    # Asegurar que la columna 'there_is_event' contenga solo valores booleanos
    valid_values = {True, False, "True", "False"}
    df['there_is_event'] = df['there_is_event'].apply(lambda x: x if x in valid_values else False)
    df['there_is_event'] = df['there_is_event'].replace({"True": True, "False": False})
    
    # Convertir la columna 'year' a numérica
    df['year_parsed'] = pd.to_numeric(df['year'], errors='coerce')
    df['date_processed'] = pd.to_datetime(df['date_processed'])
    # Definir tipos de datos para las columnas
    cols = {
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
        'year_parsed': float
    }
    
    # Convertir el DataFrame al tipo de datos especificado
    df = df.astype(cols)
    
    return df

def extraer_informacion_general_gemini(url, API_KEY_GEMINI):
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
            llm_result = stuff_chain.invoke({"context_str": context, "event_str": event} )

            return llm_result

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





