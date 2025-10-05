# ingest.py
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv() 

# --- Configuration ---
SOURCE_DIR = "./source_docs"
PERSIST_DIR = "./default_db"

# 1. Load Documents
print("Loading documents...")
documents = []
for filename in os.listdir(SOURCE_DIR):
    file_path = os.path.join(SOURCE_DIR, filename)
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 2. Split into Chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(texts)} chunks.")

# 3. Create Embeddings and Vector Store
# Note: Using a standard open-source embedding model for quick local setup.
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 

print("Creating vector store (This may take a minute)...")
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR 
)
vectordb.persist()
print(f"Ingestion complete. Vector store saved to {PERSIST_DIR}")