from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat", json={"query": "What is LLaMA 2?"})
    assert response.status_code == 200
    assert "response" in response.json()
