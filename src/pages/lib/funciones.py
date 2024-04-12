import pandas as pd
import os

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
import google.generativeai as genai

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


def cargar_eventos_procesados_archivo(path_file):
    if not os.path.exists(path_file):
        # static_1.warning('Archivo con la base de eventos no encontrada, se creara uno en blanco en la ruta "{}".'.format(PATH_DATA), icon="⚠️")
        cols = ['resume', 'there_is_event', 'title', 'general_title', 'date', 'year', 'description', 'country', 'city','place', 'key_words','asistants', 'status','google_title', 'google_snippet', 'google_long_description', 'google_url', 'search_criteria', 'date_processed','year_parsed']
        df_events = pd.DataFrame(columns = cols)
        df_events.to_excel(path_file, index=False)

    df_events = pd.read_excel(path_file)
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

def web_scrapper(url):
  """

  """
  # Initializing variable
  lang_request = TextRequestsWrapper()
  try:
    lang_request.get(url)
  except:
    print('ERR', 'Error scrappiing the url')
    return None
  # Initializing variable
  result = lang_request.get(url)
  # Initializing variable
  bs_result = BeautifulSoup(result, features="html.parser")
  # Calculating result
  text = bs_result.get_text()
  text = text.replace("\n", " ")
  text = text.replace("\t", " ")
  return text

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
        print("Hola")
        print(e)
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






