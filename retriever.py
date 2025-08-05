import os
import tempfile
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vectorstore directory
VECTOR_DIR = "vectorstore"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Shared embedding model
embeddings = HuggingFaceEmbeddings()

def store_docs(file):
    """
    Store document embeddings in a named FAISS vectorstore.
    """
    filename = file.name.replace(" ", "_").replace(".pdf", "")
    index_path = os.path.join(VECTOR_DIR, filename)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    # Load and split document
    loader = PDFPlumberLoader(tmp_file_path) 
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    # Create vectorstore and save
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)

    return {"status": "stored", "documents": len(chunks), "index_name": filename}

def get_similar_docs(query, index_name):
    """
    Retrieve top-k similar chunks from a specific document vectorstore.
    """
    index_path = os.path.join(VECTOR_DIR, index_name)
    if not os.path.exists(index_path):
        return [f"❌ No index found for '{index_name}'"]

    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)

    return docs

def reset_vectorstore():
    """
    Clear all stored vectorstore indexes.
    """
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
        os.makedirs(VECTOR_DIR)

def list_all_indexes():
    """
    List all available document index names.
    """
    return [name for name in os.listdir(VECTOR_DIR) if os.path.isdir(os.path.join(VECTOR_DIR, name))]
