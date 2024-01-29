"""Chain that calls SerpAPI.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
import os
import sys
from typing import Any, Dict, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Extra, Field, root_validator

from langchain.utils import get_from_dict_or_env


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _get_default_params() -> dict:
    return {
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "num":1
    }


def process_response(res: dict) -> str:
    """Process response from SerpAPI."""
    if "error" in res.keys():
        raise ValueError(f"Got error from SerpAPI: {res['error']}")
    if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
        toret = res["answer_box"]["answer"]
    elif "product_results" in res.keys():
        '''
        temp = res["product_results"]
        if('title' in temp):
            toret['title'] = temp['title']
        if('prices' in temp):
            toret['prices'] = temp['prices']
        if('reviews' in temp):
            toret['reviews'] = temp['reviews']
        if('rating' in temp):
            toret['reviews'] = temp['reviews']
        '''
        toret = res["product_results"]
    elif "shopping_results" in res.keys():
        toret = res["shopping_results"][0]["product_id"]
    else:
        toret = "No good search result found"
    return toret


class SerpAPIWrapper(BaseModel):
    """Wrapper around SerpAPI.

    To use, you should have the ``google-search-results`` python package installed,
    and the environment variable ``SERPAPI_API_KEY`` set with your API key, or pass
    `serpapi_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain import SerpAPIWrapper
            serpapi = SerpAPIWrapper()
    """

    search_engine: Any  #: :meta private:
    params: dict = Field(default_factory=_get_default_params)
    serpapi_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        serpapi_api_key = get_from_dict_or_env(
            values, "serpapi_api_key", "SERPAPI_API_KEY"
        )
        values["serpapi_api_key"] = serpapi_api_key
        try:
            from serpapi import GoogleSearch

            values["search_engine"] = GoogleSearch
        except ImportError:
            raise ValueError(
                "Could not import serpapi python package. "
                "Please it install it with `pip install google-search-results`."
            )
        return values

    async def arun(self, params:dict) -> str:
        """Use aiohttp to run query through SerpAPI and parse result."""

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            params = self.get_params(params)
            params["source"] = "python"
            if self.serpapi_api_key:
                params["serp_api_key"] = self.serpapi_api_key
            params["output"] = "json"
            url = "https://serpapi.com/search"
            return url, params

        url, params = construct_url_and_params()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get(url, params=params) as response:
                res = await response.json()

        return process_response(res)

    def run(self, params:dict) -> str:
        """Run query through SerpAPI and parse result."""
        params = self.get_params(params)
        with HiddenPrints():
            search = self.search_engine(params)
            res = search.get_dict()
        return process_response(res)

    def get_params(self, _params:dict) -> Dict[str, str]:
        """Get parameters for SerpAPI."""
        _params["api_key"] = self.serpapi_api_key
        params = {**self.params, **_params}
        return params


# For backwards compatibility

SerpAPIChain = SerpAPIWrapper