import os
from decouple import config
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import AzureChatOpenAI
from sales_training.sales_roleplaying import ScenarioRetrieval, SalesGPT, conversation_stages

# Create a Config object
config = config()

# Load the .env file
config.read('.env')

# Access the environment variables

def load_llm(type: str, temperature: int):
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
    config = config()
    
    if type == 'AZURE':
        llm = AzureChatOpenAI(
            openai_api_base=config.get('OPENAI_API_BASE'),
            openai_api_version=config.get('OPENAI_API_VERSION'),
            deployment_name=config.get('DEPLOYMENT_NAME'),
            openai_api_key=config.get('OPENAI_API_KEY'),
            openai_api_type=config.get('OPENAI_API_TYPE'),
            temperature=temperature
        )
        return llm
    elif type == 'AZURE_2':
        llm = AzureChatOpenAI(
            openai_api_base=config.get('OPENAI_API_BASE_2'),
            openai_api_version=config.get('OPENAI_API_VERSION_2'),
            deployment_name=config.get('DEPLOYMENT_NAME_2'),
            openai_api_key=config.get('OPENAI_API_KEY_2'),
            openai_api_type=config.get('OPENAI_API_TYPE_2'),
            temperature=temperature
        )
        return llm
    else:
        raise ValueError("Invalid 'type' parameter. Supported type is 'AZURE'.")

def return_salesGPT_instance(llm, product, template):

    config = dict(
    product=product,
    template=template,
    conversation_stage=conversation_stages.get(
            "1",
            "Introduction: Start the conversation by introducing yourself and your requirments. Briefly talk about what tech product you want to buy.",
        ),
    conversation_history="Hello, this is Gautam. How are you doing today? <END_OF_TURN>\nUser: I am well, howe are you?<END_OF_TURN>"
    )

    customer_agent = SalesGPT.from_llm(llm, verbose=False, **config)

    return customer_agent

def return_salesGPT_instance(llm, product, template):
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
    template=template,
    conversation_stage=conversation_stages.get(
            "1",
            "Introduction: Start the conversation by introducing yourself and your requirments. Briefly talk about what tech product you want to buy.",
        ),
    conversation_history="Hello, this is Gautam. How are you doing today? <END_OF_TURN>\nUser: I am well, howe are you?<END_OF_TURN>"
    )

    customer_agent = SalesGPT.from_llm(llm, verbose=False, **config)

    return customer_agent

if __name__ == "__main__":
    None    