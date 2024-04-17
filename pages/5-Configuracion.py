import streamlit as st
import datetime as dt
import os, toml, requests
import pandas as pd
from pages.lib.funciones import cargar_eventos_procesados_archivo, filtrar_df, cargar_contraseñas
from pages.lib.funciones_mongo import mdb_cargar_eventos_procesados_db
from pages.lib.funciones_snowflake import sf_cargar_eventos_procesados_db
from menu import menu
import plotly.express as px
from pages.lib.funciones import cargar_configuracion,  actualizar_configuracion


# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'
FN_KEYW = 'db_eventos_keyw.csv'
FN_EVENTS = 'events_data.xlsx'
FN_KEYW_JSON = 'app_config.json'
ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"

MODELS_DICT = {'Gemini':0, 'GROG-LLAMA2':1}
DB_DICT = {'MongoDB':0, 'Snowflake':1}
PERIODO_DICT = {"Sin restriccion" : 0, "Ultimo año":1, "Ultimo mes":2, "Ultima semana":3}
ORDEN_DICT = {"Sin orden":0, "Mas Recientes":1, "Los dos metodos":2}

#


menu()
st.image(PATH_IMG + "header_rio.jpg")

config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)

st.header("Configuracion por defecto")
col1_conf, col2_conf = st.columns([4,4])
col1_conf.markdown("***Modelo LLM*** ")
radio_modelo = col1_conf.radio(
                        "Seleccione un modelo 👉",
                        key="model",
                        options=["Gemini", "GROG-LLAMA2"],
                        index= MODELS_DICT[config['modelo']],
                        horizontal = True
                    )
col1_conf.markdown("***Numero de Paginas a buscar en google por Criterio*** ")
config['paginas'] = col1_conf.radio(
                "Seleccione numero de paginas 👉",
                key="pages",
                options=[1, 2, 3, 4, 5],
                index= config['paginas']-1,
                horizontal = True
            )
col1_conf.markdown("***Base de datos*** ")
config['base_datos'] = col1_conf.radio(
                "Seleccionar Base de datos 👉",
                key="db",
                options=["MongoDB", "Snowflake"],
                index= DB_DICT[config['base_datos']],
                horizontal = True
            )

col2_conf.markdown("***Temporalidad Busqueda*** ")
config['periodo'] = col2_conf.radio(
                "Seleccionar Base de datos 👉",
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

st.markdown("***Criterios de Busqueda*** ")
with st.expander("Ver Criterios de Busqueda", expanded =False):
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