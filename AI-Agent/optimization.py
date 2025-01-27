from langchain_core.tools import Tool
from rag_agent import rag_agent
from weather_agent import weather_agent
from calculator import calculator_agent
from web_search_agent import web_search_agent

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
RAG_TOOL_AGENT_TOOL= Tool(
    name = 'RAG_TOOL_AGENT',
    func = rag_agent,
    description = 'this agent will answer questions related to ML and LLM'

)

WEATHER_AGENT_TOOL = Tool(
    name = 'WEATHER_AGENT',
    func = weather_agent,
    description = 'this agent will give the current weather'

)

CALCULATORL_AGENT_TOOL = Tool(
    name = 'CALCULATORL_AGENT',
    func = calculator_agent,
    description = 'this agent will calculate the sum'

)

WEB_SEARCH_AGENT_TOOL = Tool(
    name = 'WEB_SEARCH_AGENT',
    func = web_search_agent,
    description = 'this agent will search the  url link'

)

tools = [RAG_TOOL_AGENT_TOOL,WEATHER_AGENT_TOOL,CALCULATORL_AGENT_TOOL,WEB_SEARCH_AGENT_TOOL]


def orchestration_agent(query:str):
    # prompt = ''' "system",
    # "You are  a helpful AI assistant tasked to answer the questions using the tools provided",
    # "You have access to 4 tools: [RAG_TOOL_AGENT_TOOL,WEATHER_AGENT_TOOL,CALCULATORL_AGENT_TOOL,WEB_SEARCH_AGENT_TOOL] 
    # the work of the tools is as follows"
    # "RAG_TOOL_AGENT_TOOL will answer the questions regarding ML"
    # "WEATHER_AGENT_TOOL will give the current weather of the city you entered"
    # "CALCULATOR_AGENT_TOOL will add the given 2 numbers"
    # "WEB_SEARCH_AGENT_TOOL will give the answers from the url provided"
    # you will trigger RAG_TOOL_AGENT_TOOL if you get a query related to ML and LLM
    # you will trigger WEATHER_AGENT_TOOL if you get a query related temperature or weather ina given city
    # you will trigger CALCULATOR_AGENT_TOOL if you get a query about addition or sum of two numbers
    # you will trigger WEB_SEARCH_AGENT_TOOL if you get a query related the url provided


    # '''
    prompt = """
You are a helpful AI assistant tasked with answering user questions using the tools provided. 

### Tools You Have Access To:
1. **RAG_TOOL_AGENT_TOOL**:
   - Use this tool to answer questions related to Machine Learning (ML) and Large Language Models (LLMs).

2. **WEATHER_AGENT_TOOL**:
   - Use this tool to provide the current weather for a given city.

3. **CALCULATOR_AGENT_TOOL**:
   - Use this tool to calculate the sum of two numbers.

4. **WEB_SEARCH_AGENT_TOOL**:
   - Use this tool to provide answers from the content of a given URL.

### Rules for Tool Usage:
- Trigger **RAG_TOOL_AGENT_TOOL** if the query is about Machine Learning or LLMs.
- Trigger **WEATHER_AGENT_TOOL** if the query is about weather or temperature for a specific city.
- Trigger **CALCULATOR_AGENT_TOOL** if the query asks for the sum of two numbers.
- Trigger **WEB_SEARCH_AGENT_TOOL** if the query is related to retrieving information from a provided URL.

Answer each query accurately and concisely, using the appropriate tool when necessary.
"""


    grog_api = os.getenv('GROG_API_KEY')
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        groq_api_key=grog_api
    )
    promt_template = ChatPromptTemplate.from_messages([
    ('system',prompt),('human','{input}'),('placeholder','{agent_scratchpad}')
    ])
    agent = create_tool_calling_agent(llm,tools,promt_template)
    my_agent = AgentExecutor(agent = agent, tools = tools)
    result =my_agent.invoke({'input':query})
    return result


orchestration_agent('what is the weather like in Fremont?')
orchestration_agent('what is machine learning?')
orchestration_agent('https://www.w3schools.com/')
