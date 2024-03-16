
import streamlit as st
import os
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import pandas as pd
import toml
import datetime as dt

# GOOGLE_API_KEY = "AIzaSyC4NWD6EqPQ-uM4xDX3MQ-Y7fgzQ1jrxU4"  # add your GOOGLE API key here
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"

FN_KEYW = 'db_eventos_keyw.csv'
FN_EVENTS = 'events_data.xlsx'

ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"

# Configuracion de la pagina
st.set_page_config(page_title="Busqueda Automatica", page_icon=":rocket:")
st.title("Busqueda Automatica")
# Define your desired data structure.
class json_resp(BaseModel):
    resume: str = Field(description="The resume of the context in few words")
    there_is_event: str = Field(description="Defines if any asociative event is mentioned. If so answer 'Yes', if not answer 'No'")
    title: str = Field(description="The title of the event, if not sure keep blank")
    date: str = Field(description="The date of the event in format YY-MM-DD, if not sure keep blank")
    year: str = Field(description="The year of the event, if not sure keep blank")
    description: str = Field(description="The description of the event, if not sure keep blank")
    country: str = Field(description="The location of the event, if not sure keep blank")
    city: str = Field(description="The city of the event, if not sure keep blank")
    key_words: str = Field(description="Only five key words of thats describe de event, separated by comma")

@st.cache_resource
def cargar_contraseñas(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        contraseñas = toml.load(f)
    return contraseñas

def cargar_llm(GEMINI_API):

    os.environ["GOOGLE_API_KEY"] = GEMINI_API
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    return llm


def cargar_eventos_keyw():
    # Cargar el archivo con las busquedas
    
    if not os.path.exists(PATH_DATA + FN_KEYW):
        st.warning('Archivo con los criterios de busqueda no encontrado en la ruta "{}", carguelo y vuelva a ejecutar.'.format(PATH_DATA), icon="⚠️")
        return None
    else:
        df_key_w = pd.read_csv(PATH_DATA + FN_KEYW)
        return df_key_w

def cargar_eventos_procesados_archivo():
    if not os.path.exists(PATH_DATA + FN_EVENTS):
        st.warning('Archivo con la base de eventos no encontrada, se creara uno en blanco en la ruta "{}".'.format(PATH_DATA), icon="⚠️")
        cols = ['google_title', 'google_snippet', 'google_long_description', 'google_url', 'llm_event_flag', 'llm_document_title', 'llm_event_name', 'llm_location', 'llm_year', 'llm_resume', 'llm_keywords', 'key_word', 'date_processed']
        df_events = pd.DataFrame(columns = cols)
        df_events.to_excel(PATH_DATA + FN_EVENTS, index=False)

    df_events = pd.read_excel(PATH_DATA + FN_EVENTS)
    # df_events["llm_event_flag"] = df_events["llm_event_flag"].astype(bool)
    st.write(df_events)
    return df_events

def query_google_search(google_query, page, search_engine_keys):
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

  url = f"https://www.googleapis.com/customsearch/v1?key={search_engine_keys['KEY']}&cx={search_engine_keys['ID']}&q={google_query}&start={start}"
  print(url)
  try:
      # Make the GET request to the Google Custom Search API
      google_response = requests.get(url)

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

def extraer_informacion_gemini(url, API_KEY_GEMINI):
    
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    llm_prompt_template = """Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    \n{format_instructions}\n{query}\n
    """
    
    loader = WebBaseLoader(url)
    
    docs = loader.load()
    parser = JsonOutputParser(pydantic_object=json_resp)

    # To extract data from WebBaseLoader
    doc_prompt = PromptTemplate.from_template("{page_content}")
    
    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
    stuff_chain = llm_prompt | llm | parser 
    llm_result = stuff_chain.invoke({"context_str": context, "query": "Is There Any event in the document?"} )

    
    return llm_result

def json_to_df(json_dict):
    import json
    try:
        # Intenta cargar el JSON en un DataFrame

        df = pd.DataFrame([json_dict])
        return df
    except Exception as e:
        print("Error al convertir JSON a DataFrame:", e)
        return None 
    
def scrapper(model = "gemini", contraseñas = None, pages=2):
    date =  dt.datetime.today().date().strftime("%Y-%m-%d")
    latest_iteration = st.empty()
    latest_iteration.text(f'Progreso 0 %')
    bar = st.progress(0)
    df_events = cargar_eventos_procesados_archivo()
    # df_events = pd.DataFrame()
    #Cargar criterios de busqueda
    df_key_w = cargar_eventos_keyw()
    step =  int(100 / (10 * (pages -1) * len(df_key_w['Key Words'])))
    i = 0
    # Buscar Paginas asociadas a los criterios
    for key_W in df_key_w['Key Words']:
        for page in range(1, pages):
            google_query_result = query_google_search(df_key_w["Key Words"].iloc[0], page, contraseñas["api_google_search"])
            for item in google_query_result.keys():
                url = google_query_result[item]['link']
                latest_iteration.text('Progreso {} %, procesando: {}'.format(i+step, url))
                bar.progress(i+step)
                i = i+step
                try:
                    llm_result = extraer_informacion_gemini(url, contraseñas["api_gemini"]['KEY'])
                except Exception as e:
                    continue
                df_event_info = json_to_df(llm_result)
                df_event_info['google_title'] = google_query_result[item]['title']
                df_event_info['google_snippet'] = google_query_result[item]['snippet']
                df_event_info['google_long_description'] = google_query_result[item]['long_description']
                df_event_info['google_url'] = google_query_result[item]['link']
                df_event_info['key_word'] =  key_W
                df_event_info['date_processed'] =  date
                df_events = pd.concat([df_events, df_event_info])
                df_events.to_excel(PATH_DATA + "events_data.xlsx", index=False)
    return df_events
    
def main():
    contraseñas = cargar_contraseñas(ACCESS_PATH)    
    
    # google_query_result = query_google_search()
    
    # Añadir un botón a la interfaz de usuario
    if st.button("Iniciar Busqueda Automatica"):
        
        df_events = scrapper(model = "gemini", contraseñas = contraseñas, pages=2)
        st.write(df_events)
        # df_key_w = cargar_eventos_keyw()
        
        # st.write(df_key_w["Key Words"].iloc[0])
        # # df_events = cargar_eventos_procesados_archivo()
        # google_query_result = query_google_search(df_key_w["Key Words"].iloc[0], 1, contraseñas["api_google_search"])
        # url = google_query_result[1]['link']
        # st.write(google_query_result[1]['link'])
        # llm_result = extraer_informacion_gemini(url, contraseñas["api_gemini"]['KEY'])
        # st.write(llm_result)
        # df = json_to_df(llm_result)
        # st.write(df)
if __name__ == "__main__":
    main()