from codigo import traduzir_texto
from testeSumari import sumarizar_texto
from testeImagens import generate_image_from_prompt

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langchain.tools import tool

def get_response_from_google(messages):
    llm = ChatVertexAI(model_name="gemini-1.5-flash", project_id="satoriai")
    response = llm.invoke(messages)
    return response

@tool
def traducao_tool(text: str, question: str):
    """Traduz um texto para o idioma solicitado pelo usuário."""
    messages = [
        SystemMessage(content="Você é um assistente de tradução para qualquer língua."),
        HumanMessage(content=f"Texto: {text}\nIdioma solicitado: {question}")
    ]
    response = get_response_from_google(messages)
    return response

@tool
def sumarizacao_tool(text: str):
    """Faz um resumo do texto dado."""
    messages = [
        SystemMessage(content="Você é um assistente de sumarização de textos."),
        HumanMessage(content=text)
    ]
    response = get_response_from_google(messages)
    return response

@tool
def generate_image_from_prompt(prompt: str):
    """
    Gera uma imagem baseada no texto (prompt) fornecido pelo usuário.
    """
    messages = [
        SystemMessage(content="Você é um assistente que gera imagens"),
        HumanMessage(content=prompt)
    ]
    response = get_response_from_google(messages)
    return response

toolkit = [traducao_tool, sumarizacao_tool,  generate_image_from_prompt]

llm = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um sistema de programação e responderá usando as ferramentas disponíveis. "
               "Caso não tenha as ferramentas necessárias, avise ao usuário. Retorne somente a resposta."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad") 
])

agent = create_openai_tools_agent(llm=llm, tools=toolkit, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

result = agent_executor.invoke({"input": "Gere uma imagem de um cavalo correndo na praia"})
print(result["output"])
