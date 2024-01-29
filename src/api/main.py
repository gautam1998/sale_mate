# main.py
import sys
import os
sys.path.append('..')
from sales_training.sales_roleplaying import ScenarioRetrieval, SalesGPT,GenerateFeedback, conversation_stages
from sales_realtime.sales_realtime import SalesRealtime, sales_realtime_output_parser
from fastapi import FastAPI, Depends, HTTPException, Request
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()
app.salesGPT = None

def load_llm(type: str="AZURE", temperature: int=0):
    """
    Load a language model based on the specified type and configuration.

    Args:
        type (str): The type of language model to load.
            Supported values: 'AZURE'.
        temperature (int): The temperature parameter for the language model.

    Returns:
        An instance of the language model based on the specified type and configuration.

    Raises:
        ValueError: If an unsupported 'type' is provided.
    """
    
    if type == 'AZURE':
        llm = AzureChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_version=os.getenv('OPENAI_API_VERSION'),
            deployment_name=os.getenv('DEPLOYMENT_NAME'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_type=os.getenv('OPENAI_API_TYPE'),
            temperature=temperature
        )
        return llm
    else:
        raise ValueError("Invalid 'type' parameter. Supported type is 'AZURE'.")

def return_salesGPT_instance(llm, product, template, persona):
    """
    Create and return a SalesGPT instance with the specified parameters.

    Parameters:
    - llm: Your Language Model instance.
    - product: The product you want to discuss.
    - template: The conversation template to use.

    Returns:
    - customer_agent: A SalesGPT instance.
    """
    config = dict(
    product=product,
    conversation_template=template,
    persona=persona,
    conversation_stage_actions=conversation_stages.get(
            "1",
            "Introduction: Start the conversation by introducing yourself and your requirments. Briefly talk about what tech product you want to buy.",
        ),
    conversation_history=""
    )

    customer_agent = SalesGPT.from_llm(llm, verbose=False, **config)
    customer_agent.seed_agent()

    return customer_agent

@app.get("/initialize_user_agent/")
async def initialize_a(prompt: str,persona: str):
    llm = load_llm()
    scenarioRetrieval = ScenarioRetrieval(llm)
    template = scenarioRetrieval.retrieve_from_vector_store(prompt)
    app.salesGPT = return_salesGPT_instance(llm, prompt, template, persona)
    #app.salesGPT = return_salesGPT_instance(llm, prompt)
    return {"message": template}

@app.get("/chatbot")
def ask_question(text: str):
#def ask_question(text: str, salesGPT: SalesGPT = Depends(return_salesGPT_instance)):
    print(text)
    app.salesGPT.human_step(
        text
    )
    response = app.salesGPT.step()
    print("RESPONSE:",response)
    response = response.lstrip("Customer:")
    return {"assistant_response": response}

@app.get("/reset_chatbot/")
async def reset_bot():
    llm = load_llm()
    app.salesGPT = return_salesGPT_instance(llm, "", "", "")
    #app.salesGPT = return_salesGPT_instance(llm, prompt)
    return {"message":"Re init"}

@app.get("/generate_feedback/")
async def generate_feedback(template: str,chat: str):
    print("In the API call")
    llm = load_llm()
    generateFeedback = GenerateFeedback(llm, template, chat)
    feedback = generateFeedback.generate_feedback()
    print(feedback)
    return {"feedback":feedback}

@app.get("/product_specifications/")
async def generate_product_specifications(query: str):
    #llm = load_llm()

    print("QUERYY:",query)
    
    llm = AzureChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE_2"),
    openai_api_version=os.getenv('OPENAI_API_VERSION_2'),
    deployment_name=os.getenv('DEPLOYMENT_NAME_2'),
    openai_api_key=os.getenv('OPENAI_API_KEY_2'),
    openai_api_type=os.getenv('OPENAI_API_TYPE_2'),
    temperature=0
    )
    salesRealtime = SalesRealtime(llm)
    agent_executor = salesRealtime.return_agent_executor()
    response = agent_executor.run(query)
    #response = sales_realtime_output_parser(response)
    return {"specifications":response}