
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.tools import Tool
from agents import get_current_weather,web_search_agent
from rag import RAG
def Orcestration_Agent(query : str):


    weather_tool = Tool(
        name="Weather",
        func=get_current_weather,
        description="Provides current weather information for a location."
    )
    rag_tool = Tool(
        name="RAG",
        func=RAG,
        description="Retrieves documents based on a query."
    )

    web_search_tool = Tool(
        name="web page search",
        func=web_search_agent,
        description="Retrieves documents from the give url."
    )
    Orcestration_agent_prompt = '''
    You are a helpful assistant tasked with answering user questions. 
    You have access to three tools: RAG , web_search_agent and get_current_weather 
    For any questions, such as questions about current events, use the web_search tool to get information from the web.,
    For any user questions about LLM agents, use the RAG tool to get information for a vectorstore. 
    For any question asked about the weather you use get_current_weather tool to get the answer"
    '''

    prompt = ChatPromptTemplate.from_messages([("system",Orcestration_agent_prompt),
                                ("human","{input}"),
                                ("placeholder","{agent_scratchpad}")])

    tools = [rag_tool,weather_tool,web_search_tool]

    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768",api_key="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")

    search_agent = create_tool_calling_agent(llm,tools,prompt)

    search_agent_exe = AgentExecutor(agent=search_agent,tools=tools)
    results = search_agent_exe.invoke({"input":query})

    return results