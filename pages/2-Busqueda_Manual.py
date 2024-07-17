
import streamlit as st
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
from menu import menu

from pages.lib.funciones import cargar_configuracion, cargar_contraseñas  
from pages.lib.funciones import limpiar_dict_event, get_embedding_gemini, check_event_embedding_gemini, query_google_search
from pages.lib.funciones_db import check_title, insert_event_db, insert_google_url_info, check_url, insert_errors_db, actualizar_estadisticas
from pages.lib.funciones_llm import extraer_informacion_url

# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'
FN_KEYW = 'db_eventos_keyw.csv'
FN_EVENTS = 'events_data.xlsx'
FN_KEYW_JSON = 'app_config.json'
ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"
#
MODELS_DICT = {'Gemini':0, 'GROG-LLAMA2':1}
st.set_page_config(layout="wide")
menu()
st.image(PATH_IMG + "header_cocora.jpg")
st.subheader("Busqueda Manual de informacion")

tab1, tab2, tab3= st.tabs(["Por URL", "Por Evento", "Lista de URLs"])


    
class Event(BaseModel):
    title: str  = Field(description="The name of the event, dont use initials, dont use punctuation marks")
    year: Optional[str]   = Field(description="The year of the event")
    country: Optional[str] = Field(description="The location of the event")

class json_resp_events(BaseModel):
    events: List[Event] = Field(..., description="The Event details")

class eventAsist(BaseModel):
    title: str  = Field(description="The name of the event, dont use initials, dont use punctuation marks")
    participants: Optional[str]   = Field(description="The resume of the information in few words about event participation, if not information or you are not sure put None")

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

def extraer_informacion_eventos_rel_gemini(url, event, API_KEY_GEMINI):
    
    os.environ["GOOGLE_API_KEY"] = API_KEY_GEMINI
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
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
    context = web_scrapper(url)
    if context.startswith('Not Acceptable!'):
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc_prompt = PromptTemplate.from_template("{page_content}")
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
        
    tokens_size = int(model.count_tokens(str(llm_prompt) + context).total_tokens)
    if tokens_size > 30000:
        return None
    else:
        stuff_chain = llm_prompt | llm | parser
        llm_result = stuff_chain.invoke({"context_str": context, "event_str": event} )
        return llm_result

def buscar_evento_url(url, contraseñas, config):
    stats = {'ejecuciones_automaticas':0, 'ejecuciones_manueales':0, 'ejecuciones_recursivas':0, 'urls':0, 'urls_eventos':0, 'eventos_nuevos':0, 'eventos' : 0, 'consultas_gse':0}
    stats['ejecuciones_manueales'] += 1
    stats['urls'] += 1
    date =  dt.datetime.today().date().strftime("%Y-%m-%d")
    flag_evento_db = False
    try:
        event_val_result, event_info_list,tokens_size, context_words  = extraer_informacion_url(url,config['modelo'])
        
        if event_val_result != None:
            if (event_val_result.there_is_event == True or event_val_result.there_is_event == 'True') and  len(event_info_list.events) > 0 :
                if event_info_list != None:
                    for event in event_info_list.events:
                        if event.there_is_event == "True" and event.title != None:
                            stats['urls_eventos'] += 1
                            stats['eventos'] += 1
                            print("Evento encontrado: {}".format(event.title))
                            if(check_title(event.title, contraseñas, config['base_datos'])):
                                print("Evento ya encontrado por titulo")
                                event = event.__dict__
                                flag_evento_db = True
                            else:
                                print("Evento no procesado segun titulo")
                                
                                if(check_event_embedding_gemini(event, contraseñas)):
                                    flag_evento_db = True
                                    event = event.__dict__
                                    print("Evento ya encontrado por busqueda semantica")
                                else:
                                    stats['eventos_nuevos'] += 1
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
                    return None, None

            else:
                print (f"No Event: {event_val_result.there_is_event}")
                return None, None
                
            if (check_url(url, contraseñas, config['base_datos'])):
                print("URL ya guardado")
            else:    
                url_info = {'google_title': '',
                            'google_snippet':'',
                            'google_long_description':'',
                            'google_url':url}    
                url_info['_id'] = url
                url_info['criterio'] = 'recursiva'
                insert_google_url_info(url_info, contraseñas, config['base_datos'])
            
            
            status = actualizar_estadisticas(stats,contraseñas, config['base_datos'])
            return event, flag_evento_db
        else:
            print (f"No Event: {event_val_result}")
            status = actualizar_estadisticas(stats,contraseñas, config['base_datos'])
            return None, None
        
        
        
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
        return None, None
    
def buscar_evento_nombre(evento_nombre, contraseñas, config):
    
    stats = {'ejecuciones_automaticas':0, 'ejecuciones_manueales':0, 'ejecuciones_recursivas':0, 'urls':0, 'urls_eventos':0, 'eventos_nuevos':0, 'eventos' : 0, 'consultas_gse':0}
    stats['ejecuciones_manueales'] += 1

    date =  dt.datetime.today().date().strftime("%Y-%m-%d")
    events_total = []
    events_result = []
    urls = []
    event_found=False
    search_params = {
                    'q': evento_nombre,
                    'lr': 'lang_esp|lang_eng',
                    }
    google_query_result = query_google_search( 1, contraseñas["api_google_search"], search_params)
    stats['consultas_gse'] += 1
    for item in google_query_result.keys():
        stats['urls'] += 1
        url = google_query_result[item]['link']
        print('#################################')
        print(url)
        try:
            event_val_result, event_info_list,tokens_size, context_words  = extraer_informacion_url(url,config['modelo'])
            if (event_val_result.there_is_event == True or event_val_result.there_is_event == 'True') and  len(event_info_list.events) > 0 :
                stats['urls_eventos'] += 1
                if event_info_list != None:
                    for event in event_info_list.events:
                        stats['eventos'] += 1
                        if event.there_is_event == "True" and event.title != None:
                            print("Evento encontrado: {}".format(event.title))
                            urls.append(url)
                            events_total.append(event)
                            if event.title == evento_nombre:
                                event_found=True
                                break
            if event_found:
                break
        
        except Exception as e:
            traceback.print_exc()
            print(f"Error:{e}" )

    for i, event in enumerate(events_total):
        print('#################################')
        print(event.title, urls[i])
        if(check_title(event.title, contraseñas, config['base_datos'])):
            event = event.__dict__
            print("Evento ya encontrado por titulo")
        else:
            print("Evento no procesado segun titulo")
            
            if(check_event_embedding_gemini(event, contraseñas)):
                event = event.__dict__
                print("Evento ya encontrado por busqueda semantica")
            else:
                stats['eventos_nuevos'] += 1
                print(f"Evento no procesado segun Busqueda Semantica, Contexto {context_words}, tokens {tokens_size}") 
                event_text = f"{event.title}, {event.description},  {event.date}, {event.year}, {event.country}, {event.city}"   
                event = event.__dict__
                event['url'] = urls[i]
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
                    
        events_result.append(event) 
        
        if (check_url(urls[i], contraseñas, config['base_datos'])):
            print("URL ya guardado")
        else:
            print("URL Guardada")    
            url_info = {'google_title': '',
                        'google_snippet':'',
                        'google_long_description':'',
                        'google_url':urls[i]}    
            url_info['_id'] = urls[i]
            url_info['criterio'] = 'recursiva'
    
    status = actualizar_estadisticas(stats,contraseñas, config['base_datos'])

    return events_result
            
    
def main():
    contraseñas = cargar_contraseñas(ACCESS_PATH)
    config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)
    with tab1:
        col1, col2 = st.columns([2, 5])
        col1.text_input("Ingrese la url", key="url")
        col1.divider()
        iniciar = col1.button("Iniciar Busqueda", key= 'bot_por_eve')
        if iniciar:
            placeholder_1 = col2.empty()
            placeholder_2 = col2.empty()

            placeholder_1.write(f"⏳ Buscando Informacion de eventos en la URL {st.session_state.url}!!")
            llm_result, flag_evento_db = buscar_evento_url(st.session_state.url, contraseñas,config )
            if llm_result != None:
                if llm_result['there_is_event'] == "True":
                    placeholder_1.write(f"✔️ Hemos encontrado eventos en la pagina {st.session_state.url}")
                    if flag_evento_db:
                        placeholder_2.warning(f"Evento almacenado previamente en Base de datos")
                    else:
                        placeholder_2.warning(f"✔️ Evento almacenado correctamente en Base de datos")
                        
                    c_1 = col2.container(border=True)
                    with col2.expander(f"Ver detalles del Evento: **{llm_result['title']}, {llm_result['country']}, {llm_result['year']}**"):
                        event_info = f"""**- Titulo del evento:** {llm_result['title']}  
                        **- Pais del evento:** {llm_result['country']} 
                        **- Año del evento:** {llm_result['year']} 
                        **- Fecha del evento:** {llm_result['date']}  
                        **- Detalles:** {llm_result['description']}
                        """
                        st.markdown(event_info)
        
    with tab2:
        col1, col2 = st.columns([2, 5])
        placeholder_1 = col2.empty()
        placeholder_2 = col2.empty()
        col1.text_input("Ingrese el nombre del evento", key="evento_nombre")
        col1.divider()
        iniciar = col1.button("Iniciar Busqueda")
        if iniciar:
            placeholder_1.write(f"⏳ Buscando Informacion de eventos segun el nombre:  {st.session_state.evento_nombre}!!")    
            eventos = buscar_evento_nombre(st.session_state.evento_nombre, contraseñas, config)
            if len(eventos) >0:
                placeholder_1.write(f"✔️ Hemos encontrado {len(eventos)} eventos para el nombre de evento: {st.session_state.evento_nombre}")
            for evento in eventos:
                with col2.expander(f"Ver detalles del Evento: **{evento['title']}, {evento['country']}, {evento['year']}**"):
                    event_info = f"""**- Titulo del evento:** {evento['title']}  
                    **- Pais del evento:** {evento['country']} 
                    **- Año del evento:** {evento['year']} 
                    **- Fecha del evento:** {evento['date']}  
                    **- Detalles:** {evento['description']}
                    """
                    st.markdown(event_info)
        
    with tab3:    
        flag_archivo = False
        df_muestra = pd.DataFrame([{'URL':""}])
        tab3_col1, tab3_col2 = st.columns([3,5])
        
        tab3_col1.markdown("***Descargar template para la lista de URLs***", unsafe_allow_html=True)
        df_muestra.to_excel("Template_urls_manual.xlsx", index = False, sheet_name="urls")
        with open(f'Template_urls_manual.xlsx', "rb") as archivo:
            file_content = archivo.read()
            tab3_col1.download_button(
                label="Descargar",
                data=file_content,
                file_name=f'Template_urls_manual.xlsx'
            )
        
        tab3_col1.markdown("***Carga el template con lista de URLs***", unsafe_allow_html=True)

        archivo_cargado = tab3_col1.file_uploader("Importar archivo de Excel", type=["xlsx", "xls"])
        
        
        if archivo_cargado is not None:
            try:
                df = pd.read_excel(archivo_cargado, sheet_name='urls')
                if df.columns[0] != "URL" or len(df.columns) > 1:
                    tab3_col1.warning(f"Formato equivocado. Descargue el template e intentelo nuevamente.", icon="⚠️")
                else:
                    tab3_col1.warning(f"Archivo Cargado Correctamente", icon="✔️")
                    flag_archivo = True
                    with tab3_col1.expander("Ver Lista de URLs"):
                    
                        lista_url = df['URL'].values.tolist()
                        for url in lista_url:
                            st.markdown(f":arrow_forward:   {url}")
            except:
                tab3_col1.warning(f"Formato equivocado. Descargue el template e intentelo nuevamente.", icon="⚠️")
                
            
        if flag_archivo:
        
            tab3_col1.markdown("***Busqueda de Eventos URLs***", unsafe_allow_html=True)
            iniciar_lista_urls = tab3_col1.button("Iniciar Busqueda", key= 'bot_lista_urls') 
            
            if iniciar_lista_urls:
                ph_tab3_1 = tab3_col2.empty()
                ph_tab3_2 = tab3_col2.empty()
                ph_tab3_3 = tab3_col2.empty()
                ph_tab3_4 = tab3_col2.empty()
                ph_tab3_2.text(f'Progreso 0 %')
                bar = tab3_col2.progress(0)
                total_ulrs = len(lista_url)
                results_lists = []
                stats = {'ejecuciones_automaticas':0, 'ejecuciones_manueales':0, 'ejecuciones_recursivas':0, 'urls':0, 'urls_eventos':0, 'eventos_nuevos':0, 'eventos' : 0, 'consultas_gse':0}

                for i,url in enumerate(lista_url):
                    ph_tab3_1.write(f"⏳ Buscando Informacion de eventos en la URL {url}!!")
                    stats['urls'] += 1
                    llm_result, flag_evento_db = buscar_evento_url(url, contraseñas,config )
                    if llm_result != None:
                        if llm_result['there_is_event'] == "True":
                            ph_tab3_3.markdown(f"✔️ Hemos encontrado eventos en la pagina {url}")
                            stats['eventos'] += 1
                            results_lists.append(llm_result)
                            if flag_evento_db:
                                ph_tab3_4.markdown(f"Evento almacenado previamente en Base de datos")
                                
                            else:
                                ph_tab3_4.markdown(f"Evento almacenado correctamente en Base de datos")
                                stats['eventos_nuevos'] += 1
                    else:
                        ph_tab3_3.markdown(f"✖️ No se encontraron eventos en la pagina {url}")
                        
                    ph_tab3_2.text(f'Progreso {100*(i+1)/total_ulrs} % - Urls Procesadas: {i+1} - Urls Faltantes: {total_ulrs - (i+1)}')
                    bar.progress((i+1)/total_ulrs)
                
                event_info = f"""**- Total URLs Procesadas :** {stats['urls']} **- Total Eventos :** {stats['eventos']} **- Total eventos nuevos:** {stats['eventos_nuevos']} 
                        """
                tab3_col2.markdown(event_info)
                
                tab3_col2.divider()   
                c_1 = tab3_col2.container(border=True)
                
                with tab3_col2.expander(f"Ver detalles de los eventos"):
                    for llm_result in results_lists:
                        event_info = f"""**- Titulo del evento:** {llm_result['title']}  
                        **- Pais del evento:** {llm_result['country']} 
                        **- Año del evento:** {llm_result['year']} 
                        **- Fecha del evento:** {llm_result['date']}  
                        **- Detalles:** {llm_result['description'][:20]} ...
                        """
                        st.markdown(event_info) 
            
if __name__ == "__main__":
    main()