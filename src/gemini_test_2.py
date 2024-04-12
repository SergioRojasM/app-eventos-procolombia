import streamlit as st
import os, toml, requests
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY="AIzaSyC4NWD6EqPQ-uM4xDX3MQ-Y7fgzQ1jrxU4"

class json_resp(BaseModel):
    resume: str = Field(description="The resume of the context in few words")
    there_is_event: str = Field(description="Defines if any asociative event is mentioned. If so answer 'Yes', if not answer 'No'")
    title: str = Field(description="The title of the event, if not sure keep blank")
    general_title: str = Field(description="The title of the event without the version or number of the event, if not sure keep blank")
    date: str = Field(description="The date of the event in format YY-MM-DD, if not sure keep blank")
    year: str = Field(description="The year of the event, if not sure keep blank")
    description: str = Field(description="The description of the event, if not sure keep blank")
    country: str = Field(description="The location of the event, if not sure keep blank")
    city: str = Field(description="The city of the event, if not sure keep blank")
    key_words: str = Field(description="Only five key words of thats describe de event, separated by comma")

def extraer_informacion_general_gemini(url):
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    llm_prompt_template = """Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    \n{format_instructions}\n{query}\n
    """
    
    loader = WebBaseLoader(url)
    
    docs = loader.load()
    parser = JsonOutputParser(pydantic_object=json_resp)

    # print(docs.page_content)
    # To extract data from WebBaseLoader
    doc_prompt = PromptTemplate.from_template("{page_content}")

    # Realizar el query a Gemini
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["context_str", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)
    
    print(model.count_tokens(str(llm_prompt) + context))
    stuff_chain = llm_prompt | llm | parser 
    # llm_result = stuff_chain.invoke({"context_str": context, "query": "Is There Any event in the document?"} )

    
    # return llm_result

url = 'https://isprm.org/events/17th-isprm-world-congress-isprm-2023-cartagena-colombia/'
# url = "https://www.isprm.org/wp-content/uploads/2012/09/ISPRM-Bidding-guidelines-2022-FINAL-20171206.pdf"
print(extraer_informacion_general_gemini(url))


# loader = WebBaseLoader(url)
    
# docs = loader.load()
# print(str(docs[0]))

# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-pro')
# print(model.count_tokens(str(docs[0])))
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-pro')
# response = model.generate_content("What is the meaning of life?")
# print(response)

# print(response.prompt_feedback)
# print(model.count_tokens("What is the meaning of life?"))