
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
import traceback
from menu import menu

from pages.lib.funciones import cargar_configuracion, cargar_contraseñas, obtener_criterios_busqueda, limpiar_dict_event
from pages.lib.funciones import  check_event_embedding_gemini, get_embedding_gemini
from pages.lib.funciones import query_google_search
from pages.lib.funciones_db import  insert_event_db, insert_errors_db, insert_google_url_info
from pages.lib.funciones_db import check_title, check_url 
from pages.lib.funciones import cargar_configuracion,  actualizar_configuracion
from pages.lib.funciones_llm import extraer_informacion_url

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
MODELS_DICT = {'Gemini':0, 'GROQ-LLAMA2':1}
DB_DICT = {'MongoDB':0, 'Snowflake':1}
PERIODO_DICT = {"Sin restriccion" : 0, "Ultimo año":1, "Ultimo mes":2, "Ultima semana":3}
ORDEN_DICT = {"Sin orden":0, "Mas Recientes":1, "Los dos metodos":2}

                        
pd.set_option('future.no_silent_downcasting', True)
# Configuracion de la pagina
st.set_page_config(page_title="Busqueda Automatica", page_icon=":rocket:",layout="wide")
st.image(PATH_IMG + "header_verde.jpg")
st.subheader("Busqueda de Eventos de Turismo")
menu()
tab1, tab2= st.tabs(["Configuración", "Busqueda Automatica"])
tab2_col1, tab2_col2 = tab2.columns([2, 5])
static_0 = tab2_col2.empty()
static_1 = tab2_col2.empty()
static_2 = tab2_col2.empty()
static_3 = tab2_col2.empty()
# Define your desired data structure.


def buscar_eventos(contraseñas = None, pages=2, list_key_w= None, config = {}):

    date =  dt.datetime.today().date().strftime("%Y-%m-%d")
    latest_iteration = tab2_col2.empty()
 
    df_events_busqueda = pd.DataFrame()
    df_errores_busqueda = pd.DataFrame()
    step =  1 / (10 * (pages) * len(list_key_w))
    static_3.text(f'Progreso 0 %')
    bar = tab2_col2.progress(0)
    i = 0
    stats = {'urls':0, 'urls_eventos':0, 'urls_eventos_nuevos':0, 'eventos' : 0}
    
    # Buscar Paginas asociadas a los criterios
    for key_W in list_key_w:
        print(key_W)
        for page in range(1, pages+1):
            google_query_result = query_google_search( page, contraseñas["api_google_search"], key_W)
            for item in google_query_result.keys():
                stats['urls'] += 1
                url = google_query_result[item]['link']
                print("###############################################################")
                print(url)
                if url.endswith('.pdf'):
                    print("URL es un PDF")
                    continue
                bar.progress(i+step)
                i = i+step
                static_1.markdown('**Criterio:** {}'.format(key_W['exactTerms']))
                static_2.markdown('**Link**: {}'.format(url))
                static_3.markdown('**Progreso:** {} %'.format(round(i*100,0)))
                if (check_url(url, contraseñas, config['base_datos'])):
                        
                    print("URL Ya Procesado")
                    continue
                else:
                    try:
                        print("URL No Procesado")
                        event_val_result, event_info_list,tokens_size, context_words  = extraer_informacion_url(url, config['modelo'])
                        
                        
                        if (event_val_result.there_is_event == True or event_val_result.there_is_event == 'True') and  len(event_info_list.events) > 0 :
                            
                            stats['urls_eventos'] += 1
                            if event_info_list != None:
                                stats['eventos'] += 1
                                for event in event_info_list.events:
                                    if event.there_is_event == "True" and event.title != None:
                                        print("Evento encontrado: {}".format(event.title))
                                        if(check_title(event.title, contraseñas, config['base_datos'])):
                                            print("Evento ya encontrado por titulo")
                                        else:
                                            print("Evento no procesado segun titulo")
                                            
                                            if(check_event_embedding_gemini(event, contraseñas)):
                                                print("Evento ya encontrado por busqueda semantica")
                                            else:
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
                        else:
                            print (f"No Event: {event_val_result.there_is_event}")
                            
                        df_google_info = pd.DataFrame([google_query_result[item]])
                        df_google_info = df_google_info.rename(columns={'title':'google_title',
                                                                        'snippet':'google_snippet',
                                                                        'long_description': 'google_long_description',
                                                                        'link':'google_url'})
                        df_google_info['_id'] = url
                        df_google_info['criterio'] = key_W
                        google_info = df_google_info.to_dict(orient='records')
                        insert_google_url_info(google_info, contraseñas, config['base_datos'])
                        
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
    static_1.markdown('**URLs Procesadas:** {}'.format(stats['urls']))
    static_2.markdown('**URLs Con Eventos:** {}'.format(stats['urls_eventos']))
    static_2.markdown('**Eventos Nuevos encontrados:** {}'.format(stats['eventos']))
    
    return df_events_busqueda


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
        
        st.header("Configuracion Busqueda Automatica")
        st.subheader("Configuracion General")
        st.divider()
        col1_conf, col2_conf = st.columns([4,4])
        col1_conf.markdown("***Numero de Paginas a buscar en google por Criterio*** ")
        config['paginas'] = col1_conf.radio(
                        "Seleccione numero de paginas 👉",
                        key="pages",
                        options=[1, 2, 3, 4, 5],
                        index= config['paginas']-1,
                        horizontal = True
                    )

        col2_conf.markdown("***Temporalidad Busqueda*** ")
        config['periodo'] = col2_conf.radio(
                        "Seleccionar periodo de busqueda 👉",
                        key="periodo",
                        options=["Sin restriccion", "Ultimo año", "Ultimo mes", "Ultima semana"],
                        index= PERIODO_DICT[config['periodo']],
                        horizontal = True
                    )
        config['orden'] = col2_conf.radio(
                        "Seleccionar orden de busqueda 👉",
                        key="orden",
                        options=["Sin orden", "Mas Recientes", "Los dos metodos"],
                        index= ORDEN_DICT[config['orden']],
                        horizontal = True
                    )

        bot_act_conf =  col2_conf.button("Actualizar Configuracion")
        if bot_act_conf:    
            actualizar_configuracion(config)
            st.rerun()
            
        st.subheader("Configuracion Criterios de Busqueda")
        st.divider()
        st.markdown("A continuacion puedes ver y modificar la configuracion asociada a los criterios  que se tendran en cuenta en el modo de busqueda automatica.")
                
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
            if col3.button("Actualizar configuracion", key="add_key"):
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
                st.rerun()
                #main()
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
            config = cargar_configuracion(PATH_DATA + FN_KEYW_JSON)
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
                    st.rerun()
                
        st.markdown("Resumen de criterios a utilizar en Google search")
        st.markdown(f" ***Criterios de Busqueda:*** {len(criterios)}" + f" ***Paginas por criterio:*** {config['paginas']}" + f" ***Total de Busquedas en Google Search:*** {config['paginas']* len(criterios)}")

        with st.expander("Ver Detalles", expanded =False):
            config = cargar_configuracion(PATH_DATA + FN_KEYW_JSON)
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
                    
            
            for i, criterio in enumerate(criterios):
                if criterio['dateRestrict'] == 'y[1]':
                    periodo = 'Ultimo año'
                elif criterio['dateRestrict'] == 'm[1]':
                    periodo = 'Ultimo mes'
                elif criterio['dateRestrict'] == 'w[1]':
                    periodo = 'Ultima semana'
                else :
                    periodo = 'Ultimos 10 años'
                    
                if "sort" in criterio.keys():
                    
                    if criterio['sort'] == 'date':
                        orden = "Mas recientes"
                else:
                    orden = "Ninguno"
                st.markdown(f"  ***Busqueda******{i+1}:***")
                st.markdown(f"***Criterio:***  {criterio['q']}, ***Idioma:***  {criterio['lr']}, ***Periodo:***  {periodo} ***Orden:***  {orden}")
                
        
    
    with tab2:
                                            
        tab2_col1.markdown("**Busqueda Automatica**")
        tab2_col1.markdown("***Configuracion***")
        tab2_col1.markdown(f"- Criterios de Busqueda: {len(criterios)}")
        tab2_col1.markdown(f"- Paginas por criterio: {config['paginas']}")
        tab2_col1.markdown(f"- Periodo: {config['periodo']}")
        tab2_col1.markdown(f"- Orden: {config['orden']}")
        tab2_col1.markdown(f"- Total de Busquedas en Google: {config['paginas']* len(criterios)}")

        iniciar_busqueda = tab2_col1.button("Iniciar Busqueda Automatica")
        if iniciar_busqueda:
            static_0.write(f"⏳ Buscando Informacion de eventos!!") 
            df_events = buscar_eventos(contraseñas, pages=config['paginas'], list_key_w= criterios, config= config)
            static_0.write(f"✔️ Hemos finalizado la busqueda de eventos ")   
            # with st.expander("Ver Resultados Encontrados:"):
            #     with st.container():
            #         st.write("***Eventos encontrados:***")
            #         st.dataframe(df_events, use_container_width=True, hide_index  = True)
            
        st.divider()    
        
        with st.expander("Ver detalles de busquedas en Google", expanded =False):
            config = cargar_configuracion(PATH_DATA + FN_KEYW_JSON)
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
                    
            
            for i, criterio in enumerate(criterios):
                if criterio['dateRestrict'] == 'y[1]':
                    periodo = 'Ultimo año'
                elif criterio['dateRestrict'] == 'm[1]':
                    periodo = 'Ultimo mes'
                elif criterio['dateRestrict'] == 'w[1]':
                    periodo = 'Ultima semana'
                else :
                    periodo = 'Ultimos 10 años'
                    
                if "sort" in criterio.keys():
                    
                    if criterio['sort'] == 'date':
                        orden = "Mas recientes"
                else:
                    orden = "Ninguno"
                st.markdown(f"  ***Busqueda******{i+1}:***")
                st.markdown(f"***Criterio:***  {criterio['q']}, ***Idioma:***  {criterio['lr']}, ***Periodo:***  {periodo} ***Orden:***  {orden}")  
if __name__ == "__main__":
    main()