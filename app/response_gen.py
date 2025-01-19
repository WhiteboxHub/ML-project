# from app.llama_model import load_llama2_model
# from app.rag_retriever import retrieve_context, create_faiss_index
# import torch

# documents = [
#     "My name is jashuva and I am machine learning engineer",
#     "R15 is one of the best bike"
#     "Gen AI is getting popular day by day"
# ]
# index, embedding_model = create_faiss_index(documents)
# tokenizer, model = load_llama2_model()

# def generate_response(query):
#     context = retrieve_context(query, index, embedding_model, documents)
#     context_str = "\n".join(context)
#     input_text = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
#     print(f"context: {context}")
#     inputs = tokenizer(input_text, return_tensors="pt")
#     inputs = {key:val.to("mps")for key,val in inputs.items()}
#     output = model.generate(**inputs, max_new_tokens=10)
#     # print(f'this is the output: {output}')
#     return tokenizer.decode(output[0], skip_special_tokens=True)
#     #return output




from app.llama_model import load_llama2_model
from app.rag_retriever import retrieve_context, create_faiss_index
import torch

# Load documents from an external text file
def load_documents(file_path):
    with open("./scraped_data.txt", 'r') as file:
        documents = [line.strip() for line in file.readlines() if line.strip()]
    return documents

documents = load_documents("./app/jb.txt")  # External file containing documents
index, embedding_model = create_faiss_index(documents)
tokenizer, model = load_llama2_model()


def generate_response(query):
    # Retrieve the most similar context using similarity search
    context = retrieve_context(query, index, embedding_model, documents)
    context_str = "\n".join(context)
    print(f"Retrieved context: {context}")  # Debugging

    # Prepare input text with the retrieved context
    input_text = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
    
    # Tokenize and move inputs to the model's device
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device  # Dynamically determine model's device
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to model's device

    # Generate response
    try:
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error during generation: {e}")
        return "An error occurred during response generation."
