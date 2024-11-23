import sys
import codecs
from google.cloud import aiplatform
from dotenv import load_dotenv
import os
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
load_dotenv()

aiplatform.init(project='satoriai', location='us-central1')

modelo = ChatVertexAI(
    model="gemini-1.5-flash",
    project_id="satoriai"
)
parser = StrOutputParser()

prompt_sumarizacao = ChatPromptTemplate.from_messages([
    SystemMessage(content="Resuma o texto a seguir:"),
    HumanMessage(content="{texto}")
])

def sumarizar_texto(texto):
    mensagens = prompt_sumarizacao.format_messages(texto=texto)
    
    resposta = modelo(mensagens)
    
    return parser.parse(resposta)
