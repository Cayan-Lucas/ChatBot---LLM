import streamlit as st
from langchain.agents import AgentExecutor
from agents import agent_executor 
from langchain.prompts import ChatPromptTemplate
from diffusers import StableDiffusionPipeline
import torch


st.set_page_config(page_title="Satori AI", layout="centered", initial_sidebar_state="expanded")


def pagina_satori():
    st.title("Olá")
    st.write("Como posso ajudar?")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def get_agent_response(user_input):
        result = agent_executor.invoke({"input": user_input})
        return result["output"]

    user_input = st.text_input("Pergunta:", key="input")

    if user_input:
        response = get_agent_response(user_input)

        st.session_state.messages.append((user_input, response))
        user_input = "" 

    for i, (user_msg, bot_msg) in enumerate(st.session_state.messages):
        st.write(f"**Usuário:** {user_msg}")
        st.write(f"**Chatbot:** {bot_msg}")

    if st.button("Ir para Gerar Imagens"):
        st.session_state.pagina = "imagens"

def pagina_imagens():
    st.title("Gerador de Imagens com Stable Diffusion")

    def generate_image_from_prompt(prompt: str):
        access_token = "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" 

        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            use_auth_token=access_token
        )
        pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

        image = pipeline(prompt).images[0]
        return image

    prompt = st.text_input("Digite o prompt para a geração da imagem:")

    if st.button("Gerar Imagem"):
        if prompt.strip():
            with st.spinner("Gerando imagem..."):
                try:
                    image = generate_image_from_prompt(prompt)

                    st.image(image, caption="Imagem gerada pela IA", use_column_width=True)
                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")
        else:
            st.warning("Por favor, insira um prompt válido para gerar a imagem.")

    if st.button("Voltar para Satori AI"):
        st.session_state.pagina = "satori"

if "pagina" not in st.session_state:
    st.session_state.pagina = "satori"

if st.session_state.pagina == "satori":
    pagina_satori()
elif st.session_state.pagina == "imagens":
    pagina_imagens()
