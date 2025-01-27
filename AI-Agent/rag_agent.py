from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# the following is to import the environment 
from dotenv import load_dotenv
import os
# to load all the environment variables
load_dotenv() 

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai_api_keys = os.getenv('OPENAI_API_KEY')

def chunker(knowledge):
    loader = TextLoader(knowledge)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
    )
    total_split_docs = text_splitter.split_documents(docs)
    print(len(total_split_docs))
    return total_split_docs

splitted_docs = chunker('knowledgeBase.txt')
splitted_docs

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
db =   FAISS.from_documents(splitted_docs,embedding)
db.embeddings

query1 = 'what does LLM stand for?'
query1_answer = db.similarity_search(query1,k=3)
print(query1_answer)

#converting the vector db to retriever class, so that we can use it with different langchain libraries

retriever= db.as_retriever()
docs_2 = retriever.invoke(query1)
docs_2

# CREATING A RAG AGENT


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.tools import tool
from typing_extensions import Annotated

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
@tool
def rag_agent(query:Annotated[str,'the query is a string']):
    ''' 
    Agent to ask a query and get relavent answer from the knowledge base

    Parameters:
    query(str): This is a string query where we can ask the llm the answeers related to the knowledge base.

    Return:
    str: It will return a string answer.
        
            
                    '''

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # #  Initializing prompt

    prompt = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer,say NAI MALUM .

            Question: {question}

            Context: {context}

            Answer:
            """

    prompt_template = ChatPromptTemplate({prompt})

    # Initializing an LLM

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key = openai_api_keys
    )


    """This code defines a chain where input documents are first formatted,
    then passed through a prompt template,
    and finally processed by an LLM."""

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt_template
        | llm
        )
    """This code creates a parallel process:
    one retrieves the context (using a retriever),
    and the other passes the question through unchanged.
    The results are then combined and assigned to the variable `answer` using the `rag_chain_from_docs` processing chain."""

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    response = rag_chain_with_source.invoke(query)
    actual_response = response['context'][0]
    return actual_response.page_content

