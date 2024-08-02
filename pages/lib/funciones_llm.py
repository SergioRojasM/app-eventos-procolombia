import yaml, os
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser, YamlOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    GoogleGenerativeAIEmbeddings
)
from typing import List, Dict, Optional, Union
from pages.lib.funciones import cargar_configuracion, cargar_contraseñas
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
###################################################################
import time

from pages.lib.config import FN_KEYW_JSON, ACCESS_PATH, PATH_DATA
from pages.lib.funciones import web_scrapper

contraseñas = cargar_contraseñas(ACCESS_PATH)
config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)

PROMPT_IDIOMA = '''
#############

# OBJECTIVE #
Your task is to identify the languaje of text below,  delimited by triple backticks.
text: ```{}```'''

PROMPT_EVENT_VAL = '''#############

# OBJECTIVE #
Your task is to identify if any  congress, symposium, conference, 
assembly, meeting, summit or seminary event is mentioned in text below, delimited by triple 
backticks.
text: ```{}``` '''

PROMPT_EXTRACCION = '''
#############

# OBJECTIVE #
Your first task is to identify if any congress, symposium, conference, 
assembly, meeting, summit or seminary event mentioned in  text below,  delimited by triple 
backticks. 
Second, extract following information only if any evetn is identified: event_type, title, general_title, date, year, 
description, country, city, place, key_words, asistants, event_category.
  
text: ```{}``` 
'''

PROMPR_CONTEXT_GEN = '''

# CONTEXT #
A tourism company wants to extract data about associative events like congress,
symposium, conference, assembly, meeting, summit or seminary from web pages. 
Those web pages will be provided to you'''

PROMPT_STYLE_GEN = '''
#############

# STYLE #
You will use only information of the text. '''
PROMPT_TONE_GEN = '''
#############

# TONE #
Use same tone as text '''

PROMPT_RESPONSE = '''        
#############

# RESPONSE #
{}''' 

PROMPT_RESPONSE_GROQ = '''        
#############

# RESPONSE #
Output must be in JSON format''' 

os.environ["GOOGLE_API_KEY"] = contraseñas["api_gemini"]['KEY']
os.environ["GROQ_API_KEY"] = contraseñas["api_groq"]['KEY']
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def obtener_tamano_tokens(text):
    model = genai.GenerativeModel('gemini-pro')
    return int(model.count_tokens(str(text)).total_tokens)   

def consulta_llm(prompt, llm_model, temp, parser, schema):
    result = None
    tamano_token = None
    tamano_palabras = len(prompt.split())
    tamano_token = obtener_tamano_tokens(prompt)
    
    if llm_model == "GEMINI":
        try:
            
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temp, 
                                        safety_settings={
                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE})
            response = llm.invoke(prompt)
            result = parser.parse(response.content)
            result = schema(**result)
            status = "Ok"
        except Exception as e:
                status = e
        
    elif llm_model == "GROQ-LLAMA3":
        try:
            llm = ChatGroq(temperature=temp, model_name="llama3-70b-8192")
            structured_groq_llm = llm.with_structured_output(schema, method="json_mode")
            result = structured_groq_llm.invoke(prompt)
            # result = result.__dict__
            status = "Ok"
        except Exception as e:
            status = e

    elif llm_model == "GROQ-MIXTRAL":
        try:
            llm = ChatGroq(temperature=temp, model_name="mixtral-8x7b-32768")
            structured_groq_llm = llm.with_structured_output(schema, method="json_mode")
            result = structured_groq_llm.invoke(prompt)
            # result = result.__dict__
            status = "Ok"
        except Exception as e:
            status = e
    time.sleep(10)
    return status, result, tamano_token, tamano_palabras

def crear_prompt(tipo, contexto ,  formato_salida):
    
    class lenguaje(BaseModel):
        idioma: str = Field(..., description="identify the language of the text", enum = ['English', 'Spanish'])
 
    class validacion_eventos(BaseModel):
        there_is_event: str = Field(..., description="Defines if any type of the mentioned events is mentioned in the text", enum = ['True', 'False'])
    
    class datos_evento(BaseModel):
        there_is_event: str = Field(..., description="Defines if any asociative event is mentioned in the context, valid events are congress, symposium, conference, assembly, meeting, summit or seminary.", enum = ['True', 'False'])
        event_type: Optional[str] = Field('None', description="describes the event type including congress, symposium, conference, assembly, meeting, summit or seminary",
                                        enum= ['Congress','Symposium','Conference','assembly','meeting','summit','seminary', 'Other'])
        title: Optional[str] = Field('None', description="The name of the event, dont use Acronyms, dont use colon punctuation")
        general_title: Optional[str] = Field(None,description="The name of the event, dont use Acronyms, don't use colon punctuation, don't specify the version of the event")
        date: Optional[str] = Field('None', description="The Date of the event, dont use colon punctuation") 
        year: Optional[str] = Field('None', description="The year of the event, only one year. if not sure use")
        description: Optional[str] = Field(None,description="Summary of the event with details of the event")
        country: Optional[str] = Field('None',description="The location of the event, if not sure use ")
        city: Optional[str] = Field('None',description="The city of the event, if not sure use ")
        place: Optional[str] = Field(None,description="The name of the place where the event takes place, if not sure use None")
        key_words: Optional[str] = Field(None,description="Only five key words of thats describe de event, separated by comma, if not sure use None")
        asistants: Optional[str] = Field(None,description="Information about number of asistants to the event, if not sure use None")
        event_category: Optional[str] = Field('None', description="describes the category of the event", 
                                        enum= ['Medical Sciences','Science','Social Sciences','Management','Education','Law','Economics','Technology','Industry',
                                                'Culture & Ideas','Arts','Commerce','Mathematics & Statistics','Safety & Security','Sports & Leisure','Ecology & Environment',
                                                'Transport & Communication','Historical Sciences','Library & Information', 'Other'])
        
    class formato_extraccion(BaseModel):
        events: List[datos_evento] = Field(..., description="The Event details") 
 
    if tipo == "idioma":
        obj = lenguaje
        PROMPT_CONTEXT = ''
        PROMPT_OBJETIVE = PROMPT_IDIOMA.format(contexto)
        PROMPT_STYLE = ''
        PROMPT_TONE = ''
    
    if tipo == "verificacion_evento":
        obj = validacion_eventos
        PROMPT_CONTEXT = PROMPR_CONTEXT_GEN
        PROMPT_OBJETIVE = PROMPT_EVENT_VAL.format(contexto)
        PROMPT_STYLE = ''
        PROMPT_TONE = ''
    
    if tipo == "extraccion_data":
        obj = formato_extraccion
        PROMPT_CONTEXT = PROMPR_CONTEXT_GEN
        PROMPT_OBJETIVE = PROMPT_EXTRACCION.format(contexto)
        PROMPT_STYLE = PROMPT_STYLE_GEN
        PROMPT_TONE = PROMPT_TONE_GEN
        
        
    if formato_salida == "json":        
        parser = JsonOutputParser(pydantic_object=obj)
    elif formato_salida == "yaml":
        parser = YamlOutputParser(pydantic_object=obj)

    prompt_template = PROMPT_CONTEXT +"\n"+ PROMPT_OBJETIVE +"\n"+ PROMPT_STYLE +"\n"+ PROMPT_TONE +"\n"+ PROMPT_RESPONSE.format(parser.get_format_instructions())

    
    
    
    return prompt_template, parser, obj
        
def extraer_informacion_url(url, model):
    print ("Scrapping pagina WEB")
    context = web_scrapper(url)
    
    print ("Validando si hay eventos....")
    ver_evento = None
    datos_evento = None
    tamano_contexto = 0
    tokens_size = 0  
      
    if context != None:
        print(f"tamaño Palabras: {len(context.split())}" )
        if len(context.split()) < 10000:
            prompt, parser, schema = crear_prompt('verificacion_evento', context, 'json' )
            ver_evento = consulta_llm(prompt, model, 0, parser, schema)
            print(ver_evento[1])
            ver_evento = ver_evento[1]
            print ("Obteniendo informacion del evento....")
            if ver_evento.there_is_event == "True" or ver_evento.there_is_event == True:
                prompt, parser, schema = crear_prompt('extraccion_data', context, 'json' )
                status, datos_evento, tokens_size, tamano_contexto  = consulta_llm(prompt, model, 0,parser, schema)
                print(datos_evento)
        else:
            print("Extraer_info_url: Contexto Muy Grande")
    else:
        print("Extraer_info_url: Contexto None")
    return ver_evento, datos_evento, tokens_size, tamano_contexto
    
