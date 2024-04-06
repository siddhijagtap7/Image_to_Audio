from transformers import pipeline
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
import os
import requests
import streamlit as st

load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
#img2text
def img2text(url):
    image_to_text = pipeline('image-to-text',model = 'Salesforce/blip-image-captioning-base')
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return(text)



def generate_story(scenario):
    template = """
    You are a story teller:
    You can generate a short story ased on a simple narrative, the story should be more than 20 words;

    CONTEXT: {scenario}
    STORY: 
    """
    prompt = PromptTemplate(template = template,input_variables= ['scenario'])
    
    story_llm = LLMChain(llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=1), prompt=prompt)

    story = story_llm.predict(scenario = scenario)
    print(story)
    return story



#text to speech
def text_to_speech(text):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
         "inputs": text
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title='Image to Audio')
    st.header('Image to Audio')
    uploaded_file = st.file_uploader('Choose an image',type='jpg')
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text_to_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.flac")

if __name__ == '__main__':
    main()

