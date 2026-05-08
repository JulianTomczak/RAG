import hashlib
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embeddings import get_embeddings

CHROMA_DIR = "./chroma_db"


def _get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        collection_name="pdf_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def index_pdf(file_path: str, filename: str) -> int:
    file_hash = _get_file_hash(file_path)

    vector_store = get_vector_store()
    existing = vector_store.get(where={"file_hash": file_hash})
    if existing and existing["ids"]:
        return 0

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["file_hash"] = file_hash

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    vector_store.add_documents(chunks)
    return len(chunks)


def list_indexed_files():
    vector_store = get_vector_store()
    data = vector_store.get()
    seen = set()
    for meta in data.get("metadatas", []):
        src = meta.get("source", "unknown")
        if src not in seen:
            seen.add(src)
    return sorted(seen)


def delete_file_from_index(filename: str):
    vector_store = get_vector_store()
    data = vector_store.get(where={"source": filename})
    if data["ids"]:
        vector_store.delete(data["ids"])
