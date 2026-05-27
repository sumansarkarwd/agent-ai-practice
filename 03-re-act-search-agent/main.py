from pathlib import Path
from typing import List

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# tavily_client = TavilyClient()


# @tool
# def search_weather(query: str) -> str:
#     """
#     Tool that searches the internet
#     Args:
#         query: query string to search for
#     Returns:
#         resut of the search by the query
#     """
#     print(f"Searching for query {query}")
#     return tavily_client.search(query)


class Source(BaseModel):
    """Schema for the result sources"""

    url: str = Field(description="URL of the source")


class AgentReponse(BaseModel):
    """Schema for the agent final response"""

    answer: str = Field(description="Final answer from the agent")
    sources: List[Source] = Field(default_factory=List, description="List of sources")


def main():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not set! Copy .env.example to .env and add your key.")

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError(
            "TAVILY_API_KEY not set! Add it to .env (see .env.example). "
            "Get a key at https://tavily.com"
        )

    llm = ChatOpenAI(
        base_url="https://openai.vocareum.com/v1",
        model="gpt-4.1-nano",
        temperature=0,
        api_key=openai_key,
    )
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentReponse)

    response = agent.invoke(
        {"messages": HumanMessage(content="Tell me the weather in Kolkata and Tokyo")}
    )

    print(response)


if __name__ == "__main__":
    main()
