
import streamlit as st
import datetime as dt
import os, toml, requests
import pandas as pd
from pages.lib.funciones import cargar_eventos_procesados_archivo, filtrar_df, cargar_configuracion, cargar_contraseñas, actualizar_configuracion, buscar_urls_pagina
from pages.lib.funciones import extraer_informacion_general_gemini_v3, limpiar_dict_event, get_embedding_gemini, check_event_embedding_gemini
from pages.lib.funciones_db import check_title, insert_event_db, insert_google_url_info, check_url, insert_errors_db
import traceback

from menu import menu

# Definicion de rutas y constantes
PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/'
FN_KEYW = 'db_eventos_keyw.csv'
FN_EVENTS = 'events_data.xlsx'
FN_KEYW_JSON = 'app_config.json'
ACCESS_PATH = PATH_CWD + "/.scrts/access.toml"
#


menu()
st.image(PATH_IMG + "header_rio.jpg")
pd.set_option('future.no_silent_downcasting', True)
# Configuracion de la pagina

st.subheader("Busqueda Recursiva de Eventos de Turismo")
menu()
tab1, tab2= st.tabs(["Configuración", "Busqueda Recursiva"])
tab2_col1, tab2_col2 = tab2.columns([2, 5])
static_0 = tab2_col2.empty()
static_1 = tab2_col2.empty()
static_2 = tab2_col2.empty()
static_3 = tab2_col2.empty()

def buscar_eventos_recursivo(contraseñas, lista_paginas, config):
    stats = {'urls':0, 'urls_eventos':0, 'urls_eventos_nuevos':0, 'eventos' : 0}
    date =  dt.datetime.today().date().strftime("%Y-%m-%d")

    for pagina in lista_paginas:
        
        lista_urls = buscar_urls_pagina(pagina)
        step =  1 / (len(lista_urls))
        st.write(step, len(lista_urls))
        static_3.text(f'Progreso 0 %')
        bar = tab2_col2.progress(0)
        i = 0
        for url in lista_urls:
            bar.progress(i+step)
            st.write(step, i)
            i = i+step
            static_1.markdown('**Pagina:** {}'.format(pagina))
            static_2.markdown('**URL**: {}'.format(url))
            static_3.markdown('**Progreso:** {} %'.format(round(i*100,0)))
            print("###############################################################")
            print(url)
            stats['urls'] += 1
            try:
                event_val_result, event_info_list,tokens_size, context_words  = extraer_informacion_general_gemini_v3(url, contraseñas["api_gemini"]['KEY'])
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
                    
                if (check_url(url, contraseñas, config['base_datos'])):
                        
                    print("URL ya guardado")
                    continue
                else:    
                    url_info = {'google_title': '',
                                'google_snippet':'',
                                'google_long_description':'',
                                'google_url':url}    
                    url_info['_id'] = url
                    url_info['criterio'] = 'recursiva'
                    insert_google_url_info(url_info, contraseñas, config['base_datos'])
                
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
                
            
    



def main():
    config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)
    contraseñas = cargar_contraseñas(ACCESS_PATH)
    with tab1:
        tab1_recursiva_col1, tab1_recursiva_col2 = st.columns([3, 3])

        tab1_recursiva_col1.markdown("***Agregar paginas busqueda recursiva*** ")
        info_1= tab1_recursiva_col2.empty()
        add_page = tab1_recursiva_col1.checkbox('Agregar', key="add_cri")
        if add_page:

            to_add_page = tab1_recursiva_col1.text_input("Ingrese nueva pagina para busqueda recursiva:", key="add_key_pagina")
            
            if tab1_recursiva_col1.button("Actualizar configuracion", key="add_pagina"):
                config_modificada = False
                if add_page:
                    if to_add_page in config['paginas_busqueda']:
                        info_1.warning(f'"**{to_add_page}**" ya esta configurado como pagina!!', icon="⚠️")
                    else:
                        config['paginas_busqueda'].append(to_add_page)
                        config_modificada = True
                        actualizar_configuracion(config)
                        info_1.markdown(f'"✔️ Alcance **{to_add_page}**" Se adiciono a la configuracion!!!')
                st.rerun()
                    
        tab1_recursiva_col1.markdown("***Eliminar paginas de busqueda recursiva*** ")
            
        rm_page = tab1_recursiva_col1.checkbox('Eliminar', key="rm_page")
        if rm_page:
            list_rmv_pages = []

            tab1_recursiva_col1.markdown("***Paginas*** ")
            for i, criterio in enumerate(config['paginas_busqueda']):
                rm_cb_page = tab1_recursiva_col1.checkbox(criterio, key=f"cb_page{i}")
                
            if tab1_recursiva_col1.button("Actualizar Configuracion", key="rm_key"):
                for i, page in enumerate(config['paginas_busqueda']):
                    if st.session_state[f"cb_page{i}"]:
                        list_rmv_pages.append(page) 

                for item in list_rmv_pages:
                    if len(config['paginas_busqueda']) > 1:
                        config['paginas_busqueda'].remove(item)
                    else:
                        info_1.warning(f'No es posible eliminar todas las paginas configuradas!!', icon="⚠️")

                actualizar_configuracion(config)
                st.rerun()
        tab1_recursiva_col2.markdown("***Paginas incluidas*** ")
        for pagina in config['paginas_busqueda']:
            tab1_recursiva_col2.write(pagina)
    with tab2:
        tab2_col1.markdown("**Busqueda Recursiva**")
        tab2_col1.markdown("***Paginas:***")
        for pagina in config['paginas_busqueda']:
            tab2_col1.write(pagina)
        
        iniciar_busqueda = tab2_col2.button("Iniciar Busqueda Recursiva")
        if iniciar_busqueda:
            static_0.write(f"⏳ Buscando Informacion de eventos!!") 
            df_events = buscar_eventos_recursivo(contraseñas, config['paginas_busqueda'], config)
            static_0.write(f"✔️ Hemos finalizado la busqueda de eventos ")   
            
                        
    
    
    
    
    

if __name__ == "__main__":
    main()
