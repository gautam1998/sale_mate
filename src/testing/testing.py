from tools.tools_definition import CustomScenarioVectorSearchTool
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import AzureChatOpenAI
import os

llm = AzureChatOpenAI(
    openai_api_base="https://openai-smart.openai.azure.com/",
    openai_api_version="2023-07-01-preview",
    deployment_name="smart",
    openai_api_key='4282343cdf384587a875631d9f8428e9',
    openai_api_type="azure",
    temperature=0
)

agent = initialize_agent(
    [CustomScenarioVectorSearchTool()], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

res = agent.run(
    "I want to buy an iphone 14"
)

print(res)