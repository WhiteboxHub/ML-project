from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama2_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
    return tokenizer, model