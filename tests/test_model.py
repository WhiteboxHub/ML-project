from app.llama_model import load_llama2_model

def test_model_load():
    tokenizer, model = load_llama2_model()
    assert tokenizer is not None
    assert model is not None