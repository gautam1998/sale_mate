import streamlit as st
import requests

def clear_chat_history():
    st.session_state.messages = []

def export_chat_history():

    if "messages" in st.session_state.keys():
        chat_str = ""
        for chat in st.session_state.messages:
            chat = chat["role"] + ": " + chat["content"]
            chat = chat + chat + "\n"

    with open("download.txt", "w") as f:
        f.write(chat_str)

st.set_page_config(page_title="Realtime Product Specification Helper")

if "messages" not in st.session_state.keys():
    st.session_state.messages = []

with st.sidebar:

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    st.sidebar.button('Export Chat History', on_click=export_chat_history)

def generate_response(prompt_input):

    url = 'http://127.0.0.1:8000/product_specifications'
    data = {
                "query": prompt_input
          }

    response = requests.get(url, params=data)
    llm_response = None

    if response.status_code == 200:
        if response.content:
            response_json = response.json()
            llm_response = response_json['specifications']
        else:
            llm_response = "Empty response from server"
    else:
        print(f'GET request failed with status code: {response.status_code}')

    print("LLM RESPONSE",llm_response.split("\n"))
    return llm_response.split("\n")

    #return ['Product ID: 6680992682506061680', 'Title: Notebook Apple MacBook Pro 256 GB SSD M2 - New - 0194253138969', "Price: ['$1,402.99', '$929.00']", "Condition: ['New', 'Refurbished']", "Typical Price: {'low': '$1,508', 'high': '$2,386', 'shown_price': '$1,402.99 at Techinn.com'}", 'Reviews: 44202', 'Rating: 4.8', 'Features:', '- Touchscreen', '- Mac OS', '- Octa Core', '- With Retina Display', '- USB-C', '- 3.5 mm Jack', '- 60Hz', '- 2560 x 1600', '- Solid State Drive', '- Silver', '', 'Image Link: [Product Image](https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcRbqmuD1lHyheqk2lDy0XE3U17Z0Z6zMpnuqSpiZ0o1rV-7bxM&usqp=CAY)']

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        content = content.replace("<END_OF_TURN>","")
        st.write(content)

st.header('Get Product Specifications :page_with_curl:')

prompt = st.chat_input("Say something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            full_response = ''''''
            for item in response:
                placeholder = st.empty()
                full_response += item
                item = item.replace("$", "\$")
                placeholder.write(item)
                full_response += '\n'
            full_response = full_response.replace("<END_OF_TURN>","")
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)