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

# EN ESTA LIBRERIA SE DEFINEN LAS LIBRERIAS A USAR EN EL APLICATIVO

# FUNCIONES PARA MANEJO DE CONFIGURACION

def cargar_configuracion(path_file):
    """
    Carga la configuración desde un archivo JSON. Si el archivo no existe, 
    crea uno nuevo con una configuración predeterminada y luego lo carga.

    Parámetros:
    path_file (str): Ruta del archivo de configuración a cargar o crear.

    Retorna:
    dict: Un diccionario con la configuración cargada o la configuración predeterminada 
          si el archivo no existía.
    
    """
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
    """
    Actualiza el archivo de configuración json de la aplicacion con los datos proporcionados.

    Parámetros:
    configuracion (dict): Un diccionario que contiene la configuración que se 
                          desea guardar en el archivo.
    """
    with open(PATH_DATA + FN_KEYW_JSON, "w") as archivo:
            json.dump(configuracion, archivo, indent=4)

def cargar_contraseñas(nombre_archivo):
    """
    Carga las contraseñas almacenadas en los secretos de Streamlit.

    Parámetros:
    nombre_archivo (str): El nombre del archivo de contraseñas, aunque no se utiliza 
                          en la implementación actual.

    Retorna:
    dict: Un diccionario que contiene las contraseñas y otros secretos almacenados 
          en `st.secrets`.

    Nota:
    Esta función está diseñada para funcionar con la plataforma Streamlit, 
    donde `st.secrets` es una propiedad que proporciona acceso a los secretos 
    definidos en el archivo de configuración de secretos de Streamlit.
    """
    return st.secrets

def obtener_criterios_busqueda(config):
    """
    Genera una lista de parámetros de búsqueda basada en la configuración proporcionada.

    Parámetros:
    config (dict): Un diccionario de configuración que contiene los siguientes campos:
        - 'periodo' (str): El período de tiempo para restringir los resultados (e.g., 'Ultimo año', 'Ultimo mes', 'Ultima semana').
        - 'orden' (str): La forma en que se deben ordenar los resultados (e.g., 'Mas Recientes', 'Los dos metodos').
        - 'patrones_busqueda' (dict): Diccionario que define los patrones de búsqueda por idioma, con subcampos para 'alcance' y 'tipo_evento'.
        - 'lugares_busqueda' (dict): Diccionario que define los lugares de búsqueda por idioma.

    Retorna:
    list: Una lista de diccionarios, donde cada diccionario contiene parámetros de búsqueda generados a partir de la configuración. Los parámetros incluyen:
        - 'q': La consulta de búsqueda construida con alcance, tipo de evento, ubicación y idioma.
        - 'lr': El lenguaje de búsqueda (español o inglés).
        - 'exactTerms': Términos exactos para la búsqueda.
        - 'dateRestrict': Restricción de fecha basada en el período especificado.
        - 'sort': Opcionalmente, la forma de ordenar los resultados si se especifica en la configuración.

    Ejemplo:
    Si `config` tiene:
    - 'periodo' = 'Ultimo mes'
    - 'orden' = 'Mas Recientes'
    - 'patrones_busqueda' con datos para 'Esp' e 'Ing'
    - 'lugares_busqueda' con datos para 'Esp' e 'Ing'

    La función generará una lista de parámetros de búsqueda para cada combinación de patrón de búsqueda, lugar y orden, restringida al período de tiempo especificado.
    """
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

### GENERALES

def es_archivo_pdf(url):
    """
    Verifica si una URL apunta a un archivo PDF.

    Parámetros:
    url (str): La URL del archivo que se desea verificar.

    Retorna:
    bool: `True` si la URL apunta a un archivo PDF, `False` en caso contrario.

    Excepciones:
    Captura y maneja excepciones relacionadas con la solicitud HTTP. Si ocurre un error al 
    hacer la solicitud (por ejemplo, una URL inválida o problemas de red), imprime un 
    mensaje de error y retorna `False`.

    """
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
     
### FUNCIONES PARA GOOGLE SEARCH

def query_google_search(page=1, search_engine_keys=None, add_params = {}):
  """
  Consulta la API de Google Custom Search y retorna un diccionario con las URLs encontradas segun los criterios ingresados

  Parametros:
      page (int): Numero de paginas a retornar.
      search_engine_keys (dict): diccionario con las claves para la API.
      add_params (dict): Dicionario con los parametros de la busqueda
  Returns:
      (List): Lista de diccionario para cada URL encontrada
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
      # Realizar la solicitud GET a la API
      google_response = requests.get(url, params=params)
      print(google_response.url)
      # Validar si la respuesta es correcta (codigo 200)
      if google_response.status_code == 200:
          # Parsear el JSON de respuesta
          google_response_data = google_response.json()
          google_response_items = {}
          search_items = google_response_data.get("items")
          # Recorrer los resultados encontrados 
          for i, search_item in enumerate(search_items, start=1):
              try:
                  long_description = search_item["pagemap"]["metatags"][0]["og:description"]
              except KeyError:
                  long_description = "N/A"
              # Obtener el titulo de la URL
              title = search_item.get("title")
              # Obtener el recorte del resumen de la URL
              snippet = search_item.get("snippet")

              # Extraer la URL
              link = search_item.get("link")
              
              # Construir la lista de diccionarios con los elementos de interes
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
    Realiza un scrapping web en la URL proporcionada utilizando diferentes métodologias.

    Parámetros:
    url (str): La URL del sitio web que se desea revisar.

    Retorna:
    str: El texto extraído de la página web. Si ocurre un error en todos los métodos de raspado, retorna `None`.

    Descripción del Proceso:
    1. **Método LangChain:**
       - Intenta realizar una solicitud a la URL usando `TextRequestsWrapper` de LangChain.
       - Si tiene éxito, usa BeautifulSoup para extraer el texto de la respuesta HTML.
       - Si hay un error, captura la excepción y sigue con el siguiente método.

    2. **Método Requests:**
       - Si el primer método falla, realiza una solicitud HTTP GET usando la biblioteca `requests`.
       - Extrae el texto de la respuesta HTML usando BeautifulSoup.
       - Si hay un error, captura la excepción y sigue con el siguiente método.

    3. **Método WebBaseLoader:**
       - Si los dos métodos anteriores fallan o el texto extraído empieza con "Not Acceptable!", intenta usar `WebBaseLoader` para cargar el contenido.
       - Usa `PromptTemplate` para formatear el contenido de los documentos y concatenarlos en un solo texto.

    Nota:
    - El texto final extraído se limpia reemplazando ":" y ";" por comas.
    - Los métodos de raspado son intentados en orden y el proceso se detiene en el primer método exitoso.
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
    """
    Limpia y normaliza los valores en un diccionario de eventos.

    Parámetros:
    diccionario (dict): Un diccionario que contiene datos de eventos a ser limpiados

    Retorna:
    dict: El diccionario con los valores limpios y normalizados.

    Proceso de Limpieza:
    1. **Reemplazo de "None":** Reemplaza los valores de cadena "None" por `None` en el diccionario.
    2. **Reemplazo de NaN:** Reemplaza los valores `np.nan` por `None`.
    3. **Reemplazo de cadenas vacías:** Reemplaza las cadenas vacías por `None`.
    4. **Normalización de texto:** Normaliza y elimina tildes en el valor de la clave `'city'` usando la función `quitar_tildes`.
    5. **Conversión de año:** Intenta convertir el valor de la clave `'year'` a un número flotante y lo guarda en la clave `'year_parsed'`. Si la conversión falla, asigna `None`.

    Función auxiliar `quitar_tildes`:
    - Elimina las tildes de un texto y devuelve el texto sin acentos.
    - Parámetro:
        - `texto` (str): Texto del cual se eliminarán las tildes.
        - Retorno: Texto sin tildes o `None` si el texto original es `None`.
    """
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
    """
    Filtra un DataFrame basado en la selección de columnas y valores proporcionados por el usuario a través de la interfaz de Streamlit.

    Parámetros:
    df (pd.DataFrame): El DataFrame que se desea filtrar.

    Retorna:
    pd.DataFrame: El DataFrame filtrado según los criterios seleccionados por el usuario.

    Descripción del Proceso:
    1. **Mostrar Checkbox para Filtros:** Usa un checkbox de Streamlit para permitir al usuario decidir si desea aplicar filtros.
    2. **Selección de Columnas para Filtrar:** Si el usuario elige crear filtros, se presenta una interfaz para seleccionar las columnas en las que desea aplicar filtros.
    3. **Filtrado por Fecha:**
       - Para la columna `'Processing Date'`, el DataFrame se convierte a formato de fecha.
       - El usuario selecciona un rango de fechas, y el DataFrame se filtra para incluir solo las filas dentro de ese rango.
    4. **Filtrado por Valores Categóricos:**
       - Para otras columnas, el usuario selecciona valores específicos para filtrar.
       - El DataFrame se filtra para incluir solo las filas con los valores seleccionados.

    Nota:
    - La función utiliza la biblioteca Streamlit para la interfaz de usuario, por lo que se debe ejecutar en un entorno donde Streamlit esté configurado.
    - El DataFrame original no se modifica; se devuelve una copia filtrada.
    """
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
    """
    Obtiene un vector de embeddings para un texto utilizando el modelo Gemini de Google.

    Parámetros:
    text (str): El texto del cual se desea obtener el embedding.
    API_KEY_GEMINI (str): La clave de API necesaria para autenticar las solicitudes con el servicio Gemini de Google Generative AI.

    Retorna:
    vector (list): El vector de embeddings generado para el texto proporcionado.
    """
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI

    llm = ChatGoogleGenerativeAI(model= "gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="semantic_similarity")

    vector = embeddings.embed_query(text)
    return vector

def comparar_eventos_gemini(event1, event2, API_KEY_GEMINI):
    """
    Compara dos eventos utilizando el modelo Gemini de Google para evaluar su similitud.

    Parámetros:
    event1 (str): Descripción del primer evento a comparar.
    event2 (str): Descripción del segundo evento a comparar.
    API_KEY_GEMINI (str): La clave de API necesaria para autenticar las solicitudes con el servicio Gemini de Google Generative AI.

    Retorna:
    dict: Un diccionario que contiene la evaluación de similitud entre los dos eventos, con un valor entre 0 y 100.

    Descripción del Proceso:
    1. **Definición de Modelo Pydantic:**
       - Se define una clase `event_already_evaluation` que describe el grado de similitud entre los dos eventos, con un valor entre 0 y 100.
    2. **Configuración del Modelo de Generación:**
       - Se crea una instancia del modelo `ChatGoogleGenerativeAI` con el modelo "gemini-pro" y se configuran los ajustes de seguridad.
       - Se crea una instancia del modelo generativo `genai.GenerativeModel` con el modelo "gemini-pro".
    3. **Definición del Prompt Template:**
       - Se define un template de prompt que describe la tarea de evaluar si los dos eventos son iguales.
       - Se utiliza el `YamlOutputParser` para formatear la salida del modelo.
    4. **Generación de la Evaluación:**
       - Se crea un prompt basado en el template y se invoca el modelo generativo con los eventos proporcionados.
       - Se devuelve el resultado de la evaluación.
    """
    
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
    """
    Verifica si un evento dado es similar a los eventos en una base de datos utilizando embeddings y comparación de eventos.

    Parámetros:
    event_in (object): Un objeto que contiene la información del evento a verificar. Debe tener atributos: `title`, `description`, `date`, `year`, `country`, y `city`.
    contraseñas (dict): Un diccionario con las credenciales necesarias para acceder a la API de Gemini y a la base de datos MongoDB. Debe contener:
        - 'api_gemini': Un diccionario con la clave de API de Gemini bajo la clave 'KEY'.
        - 'mongo_db': Un diccionario con las credenciales de MongoDB.

    Retorna:
    bool: `True` si el evento dado es considerado similar a al menos uno de los eventos en la base de datos, `False` en caso contrario.

    Descripción del Proceso:
    1. **Generación de Embedding:**
       - Se construye una consulta a partir de los atributos del evento dado.
       - Se obtiene un vector de embeddings para esta consulta usando la función `get_embedding_gemini`.
    2. **Búsqueda en la Base de Datos:**
       - Se buscan los eventos más cercanos en la base de datos MongoDB usando la función `mdb_get_k_nearest_results`, pasando el embedding y los parámetros de conexión.
    3. **Comparación de Eventos:**
       - Para cada evento recuperado de la base de datos, se construye una consulta similar.
       - Se compara el evento dado con cada evento recuperado usando la función `comparar_eventos_gemini`.
       - Si el resultado de la comparación indica que los eventos son similares en más del 90%, se establece la bandera `flag_event` en `True`.
    4. **Retorno del Resultado:**
       - La función retorna `True` si al menos un evento en la base de datos es considerado similar, `False` de lo contrario.

    """
    query = f"{event_in.title}, {event_in.description},  {event_in.date}, {event_in.year}, {event_in.country}, {event_in.city}"
    embedding = get_embedding_gemini(query, contraseñas["api_gemini"]['KEY'])
    k_events  = mdb_get_k_nearest_results(embedding, 3, 'fct_eventos_turismo', contraseñas["mongo_db"])
    flag_event = False

    if k_events:
        for event_db in k_events:
            event_db_text = f"{event_db['title']}, {event_db['description']},  {event_db['date']}, {event_db['year']}, {event_db['country']}, {event_db['city']}"
            llm_result = comparar_eventos_gemini(event_in, event_db_text, contraseñas["api_gemini"]['KEY'])
            
            if llm_result.are_same_event>90:
                flag_event = True
                
    return flag_event

def buscar_urls_pagina(sitio_web):
    """
    Busca y extrae todas las URLs de una página web dada.

    Parámetros:
    sitio_web (str): La URL del sitio web del cual se desean extraer las URLs.

    Retorna:
    list[str]: Una lista de URLs encontradas en la página web. Las URLs son absolutas y únicas.

    Proceso:
    1. **Solicitar la Página Web:**
       - Se realiza una solicitud GET a la URL proporcionada para obtener el contenido de la página.
       - Si la respuesta es exitosa (código de estado 200), se procede al análisis.
    2. **Analizar el Contenido HTML:**
       - Se utiliza BeautifulSoup para analizar el contenido HTML de la respuesta.
       - Se buscan todos los elementos `<a>` en el HTML, que representan enlaces.
    3. **Extraer URLs:**
       - Se extrae el atributo `href` de cada enlace y se verifica si no es nulo ni un fragmento (es decir, que no comience con '#').
       - Se convierte cada URL relativa en una URL absoluta utilizando `urljoin`.
       - Se añade la URL a una lista si no está ya presente.
    4. **Retornar la Lista de URLs:**
       - Se retorna la lista de URLs únicas encontradas en la página.
    """
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