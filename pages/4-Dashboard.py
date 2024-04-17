import streamlit as st
import datetime as dt
import os, toml, requests
import pandas as pd
from pages.lib.funciones import cargar_eventos_procesados_archivo, filtrar_df, cargar_contraseñas
from pages.lib.funciones_mongo import mdb_cargar_eventos_procesados_db
from pages.lib.funciones_snowflake import sf_cargar_eventos_procesados_db
from menu import menu
import plotly.express as px

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
# Define el título y la imagen de fondo
contraseñas = cargar_contraseñas(ACCESS_PATH)
df_events_hist = mdb_cargar_eventos_procesados_db(contraseñas['mongo_db'])
df_events_hist_filter = df_events_hist[(df_events_hist['status'] == "OK") &
                                        (df_events_hist['there_is_event'] == True) &
                                        (df_events_hist['country'] == "Colombia") &
                                        ((df_events_hist['year_parsed'] >= dt.datetime.today().year-10) | (df_events_hist['year_parsed'].isna()))]

cols = ['title', 'google_url', 'country', 'city', 'year', 'date', 'description', 'date_processed']
df_events_hist_filter = df_events_hist_filter[cols]
cols_name = ['Event title', 'Event URL', 'Event Country', 'Event City', 'Event Year', 'Event Date', 'Event Description', 'Processing Date']
df_events_hist_filter.columns = cols_name
df_events_hist_filter = filtrar_df(df_events_hist_filter)
row1_col1, row1_col2 = st.columns([3,8]) 
row1_col1.metric(label="Eventos Encontrados", value=str(len(df_events_hist_filter)) + " Eventos")
# row1_col2.metric(label="Paginas Analizadas", value=str(len(df_events_hist)) + " Paginas")


row2_col1, row2_col2, row2_col3 = st.columns([8,8,8]) 
df_events_proc_date = df_events_hist_filter.groupby('Processing Date').agg(numero_eventos=('Event title', 'count')).reset_index()

# Define Colores base
blue_color = '#00508D'
light_gray = 'rgb(211, 211, 211)'
line_width = 3


# Crea la gráfica con la línea y los marcadores en azul
fig = px.line(df_events_proc_date, x="Processing Date", y="numero_eventos", title="Número de Eventos por Fecha")
fig.update_traces(line=dict(color=blue_color, width=line_width), marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Número de Eventos por Fecha",
        'y':0.9, 
        'x':0.5, 
        'xanchor': 'center', 
        'yanchor': 'top', 
        'font': dict(size=20) 
    },
    xaxis=dict(title="Fecha", gridcolor=light_gray),
    yaxis=dict(title="Número de Eventos", gridcolor=light_gray)
)

row2_col1.plotly_chart(fig, theme="streamlit", use_container_width=True)

df_events_year = df_events_hist_filter.groupby('Event Year').agg(numero_eventos=('Event title', 'count')).reset_index()
# st.dataframe(df_events_year, use_container_width=True, hide_index  = True)
fig = px.bar(df_events_year, x='Event Year', y='numero_eventos')
fig.update_traces(marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Número de Eventos por Año",
        'y':0.9, # Posición vertical del título
        'x':0.5, # Posición horizontal del título (0.5 es el centro)
        'xanchor': 'center', # Alineación horizontal del título
        'yanchor': 'top', # Alineación vertical del título
        'font': dict(size=20) # Tamaño del título
    },
    xaxis=dict(title="Año", gridcolor=light_gray),
    yaxis=dict(title="Número de Eventos", gridcolor=light_gray)
)
row2_col2.plotly_chart(fig, theme="streamlit", use_container_width=True)

# row3_col1, row3_col2 = st.columns([8,8]) 

df_events_city = df_events_hist_filter.groupby('Event City').agg(numero_eventos=('Event title', 'count')).reset_index().sort_values(by='numero_eventos', ascending=False)
fig = px.bar(df_events_city, x='Event City', y='numero_eventos')
fig.update_traces(marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Número de Eventos por Ciudad",
        'y':0.9, # Posición vertical del título
        'x':0.5, # Posición horizontal del título (0.5 es el centro)
        'xanchor': 'center', # Alineación horizontal del título
        'yanchor': 'top', # Alineación vertical del título
        'font': dict(size=20) # Tamaño del título
    },
    xaxis=dict(title="Ciudad", gridcolor=light_gray, categoryorder='total descending'),
    yaxis=dict(title="Número de Eventos", gridcolor=light_gray)
)

row2_col3.plotly_chart(fig, theme="streamlit", use_container_width=True)
# fig.show()






column_config={"Event URL": st.column_config.LinkColumn("Event URL")}
st.dataframe(df_events_hist_filter, use_container_width=True, hide_index  = True, column_config= column_config)

