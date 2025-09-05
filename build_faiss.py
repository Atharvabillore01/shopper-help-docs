from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader = TextLoader("./helpdocs/docs/index.md")
docs = loader.load()

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(chunks, embedding)
db.save_local("faiss_index")
