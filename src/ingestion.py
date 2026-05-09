import hashlib
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.repository import index_chunks, check_hash
from src.file_index import add_file


def _get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def index_pdf(file_path: str, filename: str, materia: str = "") -> int:
    file_hash = _get_file_hash(file_path)
    if check_hash(file_hash):
        return 0

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["file_hash"] = file_hash
        if materia:
            doc.metadata["materia"] = materia

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    index_chunks(chunks)
    add_file(filename, materia, len(chunks), file_hash)
    return len(chunks)
