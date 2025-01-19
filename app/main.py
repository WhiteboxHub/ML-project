from fastapi import FastAPI
from app.response_gen import generate_response

app = FastAPI()

@app.post("/chat")
async def chat(query: str):
    response = generate_response(query)
    return {"response": response}