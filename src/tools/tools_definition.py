from tools.serp_API import SerpAPIWrapper
from typing import Optional, Type
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from langchain.vectorstores import Chroma
import openai
import os
from langchain.embeddings import HuggingFaceEmbeddings
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
        """Use the tool."""
        search_wrapper = SerpAPIWrapper()

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
        """Use the tool."""
        search_wrapper = SerpAPIWrapper()
        
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

class CustomScenarioVectorSearchSchema(BaseModel):
    query: str = Field(description="should return a scenario given a user query")

class CustomScenarioVectorSearchTool(BaseTool):
    name = "custom_vector_search"
    description = "useful for when you want to search for a relevat scenario given a user query"
    args_schema: Type[CustomScenarioVectorSearchSchema] = CustomScenarioVectorSearchSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        
        # Load embedding function
        emb_model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=emb_model,
                                   cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
        )

        vector_db_path = "/Users/igautam/Documents/Quarter5/capstone/saleMate/data/processed/scenarios_db"
        vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        v = vector_db.similarity_search(query, include_metadata=True)
        return v

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")