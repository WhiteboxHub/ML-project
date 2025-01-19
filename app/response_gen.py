from app.llama_model import load_llama2_model
from app.rag_retriever import retrieve_context, create_faiss_index
import torch

documents = [
        "LLaMA 2 is a language model developed by Meta.",
        "RAG integrates retrieval and generation for better context.",
        "Meta made LLaMA 2 open-source under a specific license."
    ]
index, embedding_model = create_faiss_index(documents)
tokenizer, model = load_llama2_model()

def generate_response(query):
    context = retrieve_context(query, index, embedding_model, documents)
    context_str = "\n".join(context)
    input_text = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key:val.to("mps")for key,val in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)