from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from diffusers import StableDiffusionPipeline
import torch

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um gerador de imagens baseado em inteligência artificial."),
    ("user", "Por favor, gere uma imagem com o seguinte tema: {prompt}")
])

def generate_image_from_prompt(prompt: str):
    access_token = "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"

    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        use_auth_token=access_token
    )
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipeline(prompt).images[0]
    
    image.show()

