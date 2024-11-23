from google.cloud import aiplatform
aiplatform.init(project='satoriai', location='us-central1')

from dotenv import load_dotenv
import os
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

modelo = ChatVertexAI(
    model="gemini-1.5-flash",
    project_id="satoriai"
)
parser = StrOutputParser()

def traduzir_texto(texto, idioma="italiano"):
    prompt_traducao = ChatPromptTemplate.from_messages([
        SystemMessage(content="Traduza o texto a seguir para {idioma}"),
        HumanMessage(content="{texto}")
    ])
    
    mensagens = prompt_traducao.format_messages(idioma=idioma, texto=texto)
    
    resposta = modelo(mensagens)
    return parser.parse(resposta)
