from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from typing import Annotated, List, Tuple, Union
from RAG_AND_AGENTS.prompt import web_search_agent_prompt
@tool
def get_current_weather(city : Annotated[str, "city to get the weather for the city"]) -> str:
    """
    Get the current weather for a city.

    Parameters:
    city (str): The name of the city.

    Returns:
    str: The current temperature in the specified city.
    """
    return f"The current temperature in {city} is Sunny"


url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
from groq import Groq

@tool
def web_search_agent(url : Annotated[str, "url to retrive informatinfor a webpage"]):
    # Load documents from the URLs
    """
    
    get the information from the web page from given url

    Parameters:
    url (str): The name of the city.

    Returns:
    str: the text retrived form the url

    """
    client = Groq(
    api_key="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"
    )
    docs = WebBaseLoader(url)
    docs_list = [item for item in docs ]

    if len(docs_list) > 30:
        docs_list = docs_list[1:20]
    
    chat_prompt = web_search_agent_prompt.format(context=docs_list)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role" : "system",
                "content": chat_prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )
    return chat_completion.choices[0].message.content


