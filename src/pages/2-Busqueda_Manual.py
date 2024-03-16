
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

GOOGLE_API_KEY = "AIzaSyC4NWD6EqPQ-uM4xDX3MQ-Y7fgzQ1jrxU4"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = ChatGoogleGenerativeAI(model="gemini-pro")

st.set_page_config(page_title="Busqueda Automatica", page_icon=":rocket:")

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

def extraer_informacion(url):
    
    llm_prompt_template = """Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    \n{format_instructions}\n{query}\n
    """
    
        # Add a placeholder
    latest_iteration = st.empty()
    latest_iteration.text(f'Progreso 0 %')
    bar = st.progress(0)
    
    loader = WebBaseLoader(url)
    latest_iteration.text(f'Progreso 10 %')
    bar.progress(10)
    
    docs = loader.load()
    parser = JsonOutputParser(pydantic_object=json_resp)

    # To extract data from WebBaseLoader
    doc_prompt = PromptTemplate.from_template("{page_content}")
    
    # To query Gemini

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    latest_iteration.text(f'Progreso 50 %')
    bar.progress(50)
    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
    stuff_chain = llm_prompt | llm | parser 
    llm_result = stuff_chain.invoke({"context_str": context, "query": "Is There Any event in the document?"} )
    latest_iteration.text(f'Progreso 100 %')
    bar.progress(100)
    
    st.write(llm_result)
    
    

def main():
    st.title("Busqueda Manual de informacion por URL")
    
    st.text_input("Ingrese la url", key="url")
    # Añadir un botón a la interfaz de usuario
    if st.button("Iniciar Busqueda Automatica"):
        extraer_informacion(st.session_state.url)

if __name__ == "__main__":
    main()