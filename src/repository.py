from langchain_chroma import Chroma
from src.embeddings import get_embeddings
from src.config import CHROMA_DIR, COLLECTION_NAME

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _vector_store


def index_chunks(chunks):
    get_vector_store().add_documents(chunks)


def update_file_metadata(filename: str, new_materia: str):
    vs = get_vector_store()
    data = vs.get(where={"source": filename}, include=["metadatas"])
    if not data["ids"]:
        return
    for m in data["metadatas"]:
        if new_materia:
            m["materia"] = new_materia
        elif "materia" in m:
            del m["materia"]
    vs._collection.update(ids=data["ids"], metadatas=data["metadatas"])


def delete_file(filename: str):
    vs = get_vector_store()
    data = vs.get(where={"source": filename}, include=["metadatas"])
    if data["ids"]:
        vs.delete(data["ids"])


def delete_by_materia(materia: str):
    vs = get_vector_store()
    data = vs.get(where={"materia": materia}, include=["metadatas"])
    if data["ids"]:
        vs.delete(data["ids"])


def check_hash(file_hash: str) -> bool:
    vs = get_vector_store()
    existing = vs.get(where={"file_hash": file_hash}, include=["metadatas"])
    return bool(existing and existing["ids"])


def get_retriever(k: int = 7, materia: str = None):
    search_kwargs = {"k": k}
    if materia:
        search_kwargs["filter"] = {"materia": materia}
    return get_vector_store().as_retriever(search_kwargs=search_kwargs)
