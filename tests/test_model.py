from app.llama_model import load_llama2_model

def test_model_load():
    tokenizer, model = load_llama2_model()
    assert tokenizer is not None
    assert model is not None



# from transformers import AutoModelForCausalLM, AutoTokenizer

# def test_llama2_model():
#     model_name = "meta-llama/Llama-2-7b-hf"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#     input_text = "What is LLaMA 2?"
#     inputs = tokenizer(input_text, return_tensors="pt")
#     inputs = {key: val.to(model.device) for key, val in inputs.items()}

#     # Generate response
#     output = model.generate(**inputs, max_new_tokens=50)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     print(f"Generated Response: {response}")

# test_llama2_model()
