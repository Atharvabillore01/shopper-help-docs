import os
import asyncio
from asyncio import Queue
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler


SOURCE_MD_PATH = os.getenv("SOURCE_MD_PATH", "helpdocs/docs/index.md")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")

# ---- Streaming handler ----
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put_nowait(token)

    async def get_stream(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token

# ---- FastAPI app ----
app = FastAPI(title="Shopper Help RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Build or load FAISS at startup ----
def build_or_load_index() -> FAISS:
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.isdir(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embedding, allow_dangerous_deserialization=True)

    if not os.path.exists(SOURCE_MD_PATH):
        raise FileNotFoundError(f"Markdown not found at {SOURCE_MD_PATH}")

    loader = TextLoader(SOURCE_MD_PATH, encoding="utf-8")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    db = FAISS.from_documents(chunks, embedding)
    db.save_local(INDEX_DIR)
    return db

db: Optional[FAISS] = None

@app.on_event("startup")
async def startup_event():
    global db
    db = build_or_load_index()

# ---- Request model ----
class Question(BaseModel):
    query: str

# ---- Health ----
@app.get("/")
def root():
    return {"ok": True, "msg": "Shopper Help RAG running", "endpoints": ["/ask"]}

# ---- Ask endpoint ----
@app.post("/ask")
async def ask(question: Question):
    handler = StreamHandler()

    # Hosted model (small & free). For better quality, switch repo_id to a stronger model & set HF token.
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-small",
        temperature=0.3,
        max_length=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Optional for public models but recommended
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the following question based on the provided context.
Be informative and clear. Use Markdown (e.g., **bold**, bullet points, links),
and `[Section Name](#section-name)` for navigation when relevant.

Context:
{context}

Question:
{question}
"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

    # stream result
    async def streamer():
        # We don't actually get token-by-token from HFEndpoint; we yield the final text once.
        result = await qa.ainvoke({"query": question.query})
        text = result.get("result", "")
        yield text

    return StreamingResponse(streamer(), media_type="text/plain")
