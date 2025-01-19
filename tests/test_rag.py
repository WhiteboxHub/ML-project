from app.rag_retriever import create_faiss_index, retrieve_context

def test_retrieval():
    documents = ["Doc1", "Doc2"]
    index, embedding_model = create_faiss_index(documents)
    results = retrieve_context("Doc1", index, embedding_model, documents)
    assert len(results) > 0
    assert "Doc1" in results