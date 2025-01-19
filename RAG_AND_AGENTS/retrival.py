
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import numpy as np
class SentenceTransformerEmbeddingModel(Embeddings):
    def __init__(self,modelname : str):
        self.model = SentenceTransformer(modelname)
    
    def embed_documents(self, texts):
        return [self.model.encode(d).tolist() for d in texts]

    def embed_query(self, query):
        return self.model.encode([query])[0].tolist()



def retrive_docs(query : str):
    embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    embedding_file_location = "../faiss_data_embedding"
    My_embedding_model = SentenceTransformerEmbeddingModel(embedding_model)

    embedding_docs = FAISS.load_local(embedding_file_location,embeddings=My_embedding_model,allow_dangerous_deserialization=True)

    docs = embedding_docs.similarity_search(query,k=5)
    
    result = docs_formater(docs)
    return result

def docs_formater(docs : list):
    return "\n\n".join(f"{doc.page_content} from source {doc.metadata}" for doc in docs)
    