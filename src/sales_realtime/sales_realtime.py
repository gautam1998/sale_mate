import sys
sys.path.append('/Users/igautam/Documents/GitHub/sale_mate/src/sales_realtime/')
sys.path.append('..')
from tools_definition import CustomShoppingSearchTool,CustomProductSearchTool
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import re
from typing import Union
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
)
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import Callable

os.environ['SERPAPI_API_KEY']=os.getenv('SERPAPI_API_KEY')

llm = AzureChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE_2"),
    openai_api_version=os.getenv('OPENAI_API_VERSION_2'),
    deployment_name=os.getenv('DEPLOYMENT_NAME_2'),
    openai_api_key=os.getenv('OPENAI_API_KEY_2'),
    openai_api_type=os.getenv('OPENAI_API_TYPE_2'),
    temperature=0
)

tools = [CustomShoppingSearchTool(),CustomProductSearchTool()]

def get_tools(query):
    tools = [CustomShoppingSearchTool(),CustomProductSearchTool()]
    return tools

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

class SalesRealtime:

    def __init__(self, LLM):

        self.LLM = LLM
        self.llm = llm

        '''
        self.template = """Answer the following questions as best you can, you have access to the following tools:
        {tools}
        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        Question: {input}
        {agent_scratchpad}
        """
        '''

        self.template = """Answer the following questions as best you can, you have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Output the final answer in the following format

        Final Answer:
        <start format>
        Product ID: <product_id>
        Title: <title>
        Price: <prices>
        Condition: <conditions>
        Typical Price: <typical_prices>
        Reviews: <reviews>
        Rating: <rating>
        Features:
        <#each features>
        - <this>
        </each>

        Image Link: [Product Image]({{image_link}})
        <end format>

        Use the above format to return the putput
        Question: {input}
        {agent_scratchpad}"""

        self.prompt = CustomPromptTemplate(
            template=self.template,
            tools_getter=get_tools,
            input_variables=["input", "intermediate_steps"],
        )

        self.output_parser = CustomOutputParser()
        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt)
        self.tool_names = [tool.name for tool in tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=self.tool_names,
        )
   
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=tools, verbose=True
        )

    def return_agent_executor(self):
        return self.agent_executor

def sales_realtime_output_parser(response):

    '''
    # Accessing specific information from the response
    product_id = response['product_id']
    title = response['title']
    prices = response['prices']
    conditions = response['conditions']
    typical_prices = response['typical_prices']
    reviews = response['reviews']
    rating = response['rating']
    extensions = response['extensions']
    description = response['description']
    media = response['media']

    response_string = (
        f"Product ID: {product_id}\n"
        f"Title: {title}\n"
        f"Prices: {', '.join(prices)}\n"
        f"Conditions: {', '.join(conditions)}\n"
        f"Typical Prices: {typical_prices['shown_price']}\n"
        f"Reviews: {reviews}\n"
        f"Rating: {rating}\n"
        f"Extensions: {', '.join(extensions)}\n"
        f"Description: {description}\n"
        f"Media: {', '.join([m['link'] for m in media])}\n"
    )

    print(response_string)

    return response_string
    '''
    return response

'''
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
response = agent_executor.run("I want to buy an Iphone")
print("response",response)
sales_realtime_output_parser(response)
'''