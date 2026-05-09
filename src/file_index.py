import json
from src.config import BASE_DIR

FILE_INDEX_PATH = BASE_DIR / "file_index.json"


def _load():
    try:
        with open(FILE_INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(index):
    with open(FILE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def list_files():
    if not FILE_INDEX_PATH.exists():
        sync_from_chromadb()
    index = _load()
    result = [{"filename": k, "materia": v.get("materia", "")} for k, v in index.items()]
    return sorted(result, key=lambda x: x["filename"])


def add_file(filename: str, materia: str, chunks: int, file_hash: str):
    index = _load()
    index[filename] = {"materia": materia, "chunks": chunks, "file_hash": file_hash}
    _save(index)


def remove_file(filename: str):
    index = _load()
    index.pop(filename, None)
    _save(index)


def update_materia(filename: str, materia: str):
    index = _load()
    if filename in index:
        index[filename]["materia"] = materia
        _save(index)


def sync_from_chromadb():
    from src.repository import get_vector_store
    vs = get_vector_store()
    data = vs.get(include=["metadatas"])
    index = {}
    for meta in data.get("metadatas", []):
        src = meta.get("source", "unknown")
        if src not in index:
            index[src] = {
                "materia": meta.get("materia", ""),
                "file_hash": meta.get("file_hash", ""),
            }
    _save(index)
