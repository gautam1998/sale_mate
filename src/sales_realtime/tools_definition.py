from serp_API import SerpAPIWrapper
from typing import Optional, Type
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
import openai
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class CustomShoppingSearchSchema(BaseModel):
    query: str = Field(description="should be a search query should only be the product name")

class CustomShoppingSearchTool(BaseTool):
    name = "custom_shopping_search"
    description = "useful for when you need to search for a prodcut when you are given the product name"
    args_schema: Type[CustomShoppingSearchSchema] = CustomShoppingSearchSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        json_data = {"search_engine": "google_shopping"}
        search_wrapper = SerpAPIWrapper(**json_data)

        params={
            "engine": 'google_shopping',
            "q": query
        }

        print("PARAMS HERE:",params)

        return search_wrapper.run(params)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class CustomProductSearchSchema(BaseModel):
    query: str = Field(description="should be a search query should only be the product id , a random number in string type")

class CustomProductSearchTool(BaseTool):
    name = "custom_product_search"
    description = "useful for when you need to search for a prodcut when you are given the product id, the input might just be a random number"
    args_schema: Type[CustomProductSearchSchema] = CustomProductSearchSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        json_data = {"search_engine": "google_product"}
        search_wrapper = SerpAPIWrapper(**json_data)
        
        params={
            "engine": 'google_product',
            "product_id": query,
        }

        return search_wrapper.run(params)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

