import streamlit as st
import os
import requests
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
from transformers import pipeline
import numpy as np
import librosa
from elevenlabs import generate, play
from elevenlabs import set_api_key

set_api_key('3f9b0199578d0f95b7c29db990590675')
st.set_page_config(page_title="Sale Mate Application")
base_url = 'http://127.0.0.1:8000/'
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

st.header('Sales Simulator :speech_balloon:')

def transcribe(audio):

    file_path = 'audio.wav' 
    y, sr = librosa.load(file_path, sr=None)
    #y = np.array(audio.raw_data, dtype=np.int16)
    #sr = audio.frame_rate
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def init_chatbot(prompt,persona):

    url = 'http://127.0.0.1:8000/initialize_user_agent'
    data = {
            "prompt": prompt,
            "persona":persona
    }
    
    try:
        response = requests.get(url, params=data, timeout=30)  # Set a timeout of 30 seconds
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return -1

    if response.status_code == 200:
        print(f'Init success: {response.status_code}')
        string_template = response.content
        with st.expander("See template conversation"):
            st.write(string_template)
        return response.content
    else:
        print(f'GET request failed with status code: {response.status_code}')
        return None

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

def clear_chat_history():
    st.session_state.messages = []

def export_chat_history():

    if "messages" in st.session_state.keys():
        chat_str = ""
        for chat in st.session_state.messages:
            chat_str_temp = chat["role"] + ": " + chat["content"]
            chat_str = chat_str + chat_str_temp + "\n"

    print("CHAT HISOTRY",chat_str)
    with open("download.txt", "w") as f:
        f.write(chat_str)

with st.sidebar:

    product_input = st.sidebar.text_input("Enter the product you want to practice selling", "")
    persona_input = st.sidebar.text_input("Enter the persona/requirements of the user", "")

    if st.sidebar.button("Submit"):
        init_chatbot(product_input, persona_input)

    voice = st.checkbox('Enable Voice')
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    st.sidebar.button('Export Chat History', on_click=export_chat_history)
    audio = audiorecorder("Click to record", "Click to stop recording")
    #st.sidebar.button('Click to record', on_click=audio)
    

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        content = content.replace("<END_OF_TURN>","")
        #print("Write Happening Here")
        st.write(content)

def generate_response(prompt_input, voice):

    print('GENERATING RESPONSE',prompt_input)

    url = 'http://127.0.0.1:8000/chatbot'
    data = {
                "text": prompt_input
          }
    response = requests.get(url, params=data)
    llm_response = None
    if response.status_code == 200:
        if response.content:
            response_json = response.json()
            llm_response = response_json['assistant_response']
        else:
            llm_response = "Empty response from server"
    else:
        print(f'GET request failed with status code: {response.status_code}')
    
    #return llm_response
    
    bot_message = llm_response

    if(voice):

        bot_message = bot_message.replace("<END_OF_TURN>","")

        audio = generate(
            text=bot_message,
            voice="Bella"
        )

        play(audio)

    return bot_message

prompt = st.chat_input("Say something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if len(audio) > 0:
    #st.audio(audio.export().read()) 
    audio.export("audio.wav", format="wav") 
    text = transcribe(audio)
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.write(text)
        prompt = text

# Generate a new response if last message is not from assistant
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt,voice)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            full_response = full_response.replace("<END_OF_TURN>","")
            placeholder.markdown(full_response)
            #placeholder.markdown("full_response")
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)