import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def create_faiss_index(documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embedding_model

def retrieve_context(query, index, embedding_model, documents, k=2):
    query_embedding = embedding_model.encode([query])
    print(f'Query embeddings: {query_embedding}')
    distances, indices = index.search(query_embedding, k)
    print(f"Retrived indices: {indices}, distances: {distances}")
    return [documents[i] for i in indices[0]]