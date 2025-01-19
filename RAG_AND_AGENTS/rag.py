from groq import Groq
from retrival import retrive_docs
from prompt import RAG_PROMPT
from typing import Annotated
def RAG(query : Annotated[str, "query to ask the retrieve information tool"]):
    """
    get the information from the knowledbase

    Parameters:
    query (str): The question from the user.

    Returns:
    str: the text retrived form the knowledgebase
    """
    # Initialize Llama model and tokenizer
    client = Groq(
    api_key="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"
    )

    # RAG_PROMPT

    retrive_doc = retrive_docs(query)
    
    chat_prompt = RAG_PROMPT.format(question=query,context=retrive_doc)
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
