import streamlit as st
import os

PATH_CWD = os.getcwd()
PATH_DATA = PATH_CWD + "/src/data/"
PATH_IMG  = PATH_DATA + 'img/' 
def menu():
    # Show a navigation menu for authenticated users
    st.sidebar.image(PATH_IMG + "procolombia_grey.jpg")
    st.sidebar.divider()
    st.sidebar.page_link("app_eventos_procolombia.py", label=":gray[Pagina Paincipal]")
    st.sidebar.page_link("pages/1-Busqueda_Automatica.py", label=":gray[Busqueda Automatica]")
    st.sidebar.page_link("pages/2-Busqueda_Manual.py", label=":gray[Busqueda Manual]")
    st.sidebar.page_link("pages/3-Dashboard.py", label=":gray[Dashboard]")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.image(PATH_IMG + "min.jpg")
