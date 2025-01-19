from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI 
from langchain.agents import initialize_agent
from agentic.agent import Agent
from agentic.tools import LLMTool

# 1. Load and Prepare Data
def load_and_prepare_data(document_path):
    """
    Loads the document, splits it into chunks, and creates embeddings.

    Args:
        document_path: Path to the input document.

    Returns:
        vectorstore: FAISS vectorstore containing document embeddings.
    """
    loader = TextLoader(document_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings() 
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# 2. Initialize LLM and RetrievalQA Chain
def initialize_llm_and_qa(llm_model_name):
    """
    Initializes the LLM and creates the RetrievalQA chain.

    Args:
        llm_model_name: Name of the LLM model to use (e.g., "text-davinci-003").

    Returns:
        qa_chain: RetrievalQA chain for answering questions.
    """
    llm = OpenAI(model_name=llm_model_name) 
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )
    return qa_chain

# 3. Create Agentic AI Agent
def create_agent(qa_chain):
    """
    Creates an Agentic AI agent with LLM and RetrievalQA tools.

    Args:
        qa_chain: RetrievalQA chain for answering questions.

    Returns:
        agent: Initialized Agentic AI agent.
    """
    tools = [
        LLMTool(
            llm=llm, 
            name="llm", 
            description="A large language model, can generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way." 
        ),
        LLMTool(
            llm=qa_chain, 
            name="qa", 
            description="A tool that can answer questions based on a given text document." 
        )
    ]
    agent = Agent(name="RAG_Agent", tools=tools) 
    return agent

# 4. Run the Agent
def run_agent(agent, user_input):
    """
    Runs the agent with the given user input.

    Args:
        agent: Initialized Agentic AI agent.
        user_input: User's question or instruction.

    Returns:
        response: Agent's response.
    """
    return agent.run(user_input)

if __name__ == "__main__":
    document_path = "jb.txt"  # Replace with the actual path
    llm_model_name = "text-davinci-003"  # Replace with your LLaMA 2 model if available

    vectorstore = load_and_prepare_data(document_path)
    qa_chain = initialize_llm_and_qa(llm_model_name)
    agent = create_agent(qa_chain)

    user_input = "What is the main topic of this document?"
    response = run_agent(agent, user_input)
    print(response)