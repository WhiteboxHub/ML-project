from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from typing_extensions import Annotated
from groq import Groq

@tool
def web_search_agent(url : Annotated[str,'It is a valid webpage']):
    ''' 
    Scraping the given webpage and getting the data.

    Parameters:
    url (str) : It is a valid webpage

    Returns:
     str : It is a summarized information of the webpage
    
    '''
    try:
        loader = WebBaseLoader(url)
        coffee_text = loader.load()
        data = ' '.join(coffee_text[0].page_content.split())
        prompt = ''' 
        <Role> You are an assistant tasked with summarization of a given context. The summary should be 300-400 words length. the summary should contain impoetant
        information.Grammer is not important.</Role>
        <Context>{context}</Context>
        
            '''
        updatedPrompt = prompt.format(context=data)



        client = Groq(
        api_key="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"
        )
        
        chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": updatedPrompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        )

        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f'there is an unknown exception as follows {e}'