import streamlit as st
import datetime as dt
import os, toml, requests
import pandas as pd
from pages.lib.funciones import  cargar_contraseñas, cargar_configuracion
from pages.lib.funciones_db import leer_estadisticas

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

# Configuracion Pagina
menu()
st.image(PATH_IMG + "header_rio.jpg")
row2_col1, row2_col2, row2_col3= st.columns([8,8,8]) 

# Cargar configuracion y contraseñas
contraseñas = cargar_contraseñas(ACCESS_PATH)
config = cargar_configuracion( PATH_DATA + FN_KEYW_JSON)

# Leer informacion de estadisticas de uso de la base de datos
df_stats = leer_estadisticas('',contraseñas, config['base_datos'])


# # Define Colores base de las graficas
blue_color = '#00508D'
light_gray = 'rgb(211, 211, 211)'
line_width = 3


# Crear la gráfica para numero de ejecuciones automaticas
fig = px.line(df_stats, x="_id", y="ejecuciones_automaticas", title="Ejecuciones Automaticas", markers=True)
fig.update_traces(line=dict(color=blue_color, width=line_width), marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Ejecuciones Automaticas",
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


# Crear la gráfica para numero de ejecuciones manuales
fig = px.line(df_stats, x="_id", y="ejecuciones_manueales", title="Ejecuciones Manuales", markers=True)
fig.update_traces(line=dict(color=blue_color, width=line_width), marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Ejecuciones Manuales",
        'y':0.9, 
        'x':0.5, 
        'xanchor': 'center', 
        'yanchor': 'top', 
        'font': dict(size=20) 
    },
    xaxis=dict(title="Fecha", gridcolor=light_gray),
    yaxis=dict(title="Número de Eventos", gridcolor=light_gray)
)

row2_col2.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Crear la gráfica para numero de ejecuciones recursivas
fig = px.line(df_stats, x="_id", y="ejecuciones_recursivas", title="Ejecuciones Recursivas", markers=True)
fig.update_traces(line=dict(color=blue_color, width=line_width), marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Ejecuciones Recursivas",
        'y':0.9, 
        'x':0.5, 
        'xanchor': 'center', 
        'yanchor': 'top', 
        'font': dict(size=20) 
    },
    xaxis=dict(title="Fecha", gridcolor=light_gray),
    yaxis=dict(title="Número de Eventos", gridcolor=light_gray)
)

row2_col3.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Crear la gráfica para numero de URLs procesadas
fig = px.line(df_stats, x="_id", y="urls", title="URLs Procesadas", markers=True)
fig.update_traces(line=dict(color=blue_color, width=line_width), marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "URLs Procesadas",
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

# Crear la gráfica para numero de eventos nuevos encontrados 
fig = px.line(df_stats, x="_id", y="eventos_nuevos", title="Eventos Nuevos Encontrados", markers=True)
fig.update_traces(line=dict(color=blue_color, width=line_width), marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Eventos Nuevos Encontrados",
        'y':0.9, 
        'x':0.5, 
        'xanchor': 'center', 
        'yanchor': 'top', 
        'font': dict(size=20) 
    },
    xaxis=dict(title="Fecha", gridcolor=light_gray),
    yaxis=dict(title="Número de Eventos", gridcolor=light_gray)
)

row2_col2.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Crear la gráfica para numero de busquedas en Google search
fig = px.line(df_stats, x="_id", y="consultas_gse", title="Consultas GSE", markers=True)
fig.update_traces(line=dict(color=blue_color, width=line_width), marker=dict(color=blue_color))
fig.update_layout(
    title={
        'text': "Consultas GSE",
        'y':0.9, 
        'x':0.5, 
        'xanchor': 'center', 
        'yanchor': 'top', 
        'font': dict(size=20) 
    },
    xaxis=dict(title="Fecha", gridcolor=light_gray),
    yaxis=dict(title="Número de Eventos", gridcolor=light_gray)
)

row2_col3.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Mostrar resumen en tabla 
column_config={"_id": "Fecha",
               "ejecuciones_automaticas" : "Ejecuciones Automaticas",
               "ejecuciones_maueales" : "Ejecuciones Manuales",
               "ejecuciones_recursivas" : "Ejecuciones Recursivas",
               "urls" : "Total URLs Procesadas",
               "urls_eventos" : "Total URLs con Eventos",
               "urls_eventos" : "Total URLs con Eventos",
               "eventos" : "Eventos Encontrados",
               "eventos_nuevos" : "Eventos Nuevos Encontrados",
               "consultas_gse" : "Contultas a GSE",
               }
st.dataframe(df_stats, hide_index=True, column_config=column_config, width=10000)