# from transformers import AutoTokenizer, AutoModelForCausalLM

# def load_llama2_model():
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
#     return tokenizer, model


from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama2_model():
    # Load a smaller model (distilgpt2 for testing purposes)
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
