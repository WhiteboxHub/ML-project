import nltk
nltk.download('punket-tab')
nltk.download('punket')
from nltk.tokenize import sent_tokenize

def read_text_file(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        text = file.read()
    
    return text


def fixed_length_chunking(file_path,chunk_size):
    text = read_text_file(file_path)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks



def sentence_based_chunking(file_path):
    text = read_text_file(file_path)
    chunks = sent_tokenize(text)
    return chunks


from langchain_text_splitters import RecursiveCharacterTextSplitter

def ovelap_chunking(file_path,chunk_size):
    overlap = int((chunk_size/100) * 20)
    text = read_text_file(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=chunk_size,
    chunk_overlap=overlap,
    length_function=len,
    is_separator_regex=False,
    )
    text_chunks = text_splitter.create_documents([text])
    return text_chunks

from transformers import pipeline

def sementic_chunking(file_path, model_name='distilbert-base-uncased'):
    summarizer = pipeline('summarization', model=model_name)
    text = read_text_file(file_path)
    # Split text into large chunks first (arbitrary size)
    preliminary_chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    # Refine chunks by summarizing each
    refined_chunks = [summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                      for chunk in preliminary_chunks]
    return refined_chunks