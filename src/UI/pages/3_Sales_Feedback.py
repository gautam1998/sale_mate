import streamlit as st
import pandas as pd
import requests

def generate_feedback(string_template,string_chat):
    
    url = 'http://127.0.0.1:8000/generate_feedback'
    
    print("template HAS COME",string_template)
    print("chat HAS COME",string_chat)

    data = {
            "template": "",
            "chat": string_chat
    }

    response = requests.get(url, params=data)
    feedback = None
    if response.status_code == 200:
        if response.content:
            response_json = response.json()
            feedback = response_json['feedback']
            print("FEEDBACK HAS COME",feedback)
        else:
            feedback = "Empty response from server"
    else:
        print(f'GET request failed with status code: {response.status_code}')
    
    return feedback



st.header("Evaluate Your Sales Conversations :pencil:")

tab1, tab2, tab3 = st.tabs(["Upload","Conversation", "Feedback"])

str_conversation = 'No Conversation Uploaded'
feedback = 'No Conversation Uploaded'

with tab1:

    uploaded_file = st.file_uploader("Add a sales conversation !")
    str_conversation = ''
    if uploaded_file:
        for line in uploaded_file:
            str_conversation = str_conversation + str(line.decode('utf-8'))
            str_conversation = str_conversation + '\n'
    feedback = generate_feedback("",str_conversation)

with tab2:
    if(str_conversation == 'No Conversation Uploaded'):
        st.write(str_conversation)
    else:
        str_conversation = str_conversation.split("\n")
        for conversation in str_conversation:
            conversation = conversation.replace("<END_OF_TURN>","")
            st.write(conversation)

with tab3:
    if(feedback == 'No Conversation Uploaded'):
        st.write(feedback)
    else:
        feedback = feedback.split("\n")
        scoring = {}
        text_feedback = None
        for feed in feedback:
            if(':' in feed):
                temp = feed.split(":")
                scoring[temp[0]] = temp[1]
                #scoring.append(feed)
            else:
                if(len(feed) > 0):
                    text_feedback = feed

        st.dataframe(pd.DataFrame(scoring.items(),columns=['Criteria','Score']))
        st.write(feed)

