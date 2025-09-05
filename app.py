from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from shopper_help_rag import qa  # your RAG pipeline

app = FastAPI()

# Allow frontend (GitHub Pages, Hugging Face Space, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your GH Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: Query):
    answer = qa.run(request.query)
    return {"answer": answer}
