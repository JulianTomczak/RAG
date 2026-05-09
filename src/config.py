from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "pdf_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 7
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.3
MODEL_MAX_TOKENS = 128000

DEFAULT_SUBJECT_PROMPT = """Respondé la pregunta de forma clara, precisa y bien estructurada.

Instrucciones:
- Basate prioritariamente en el contexto proporcionado a continuación.
- Usá un lenguaje preciso y la terminología apropiada al tema.
- Si el contexto no alcanza para responder completamente, indicálo en lugar de inventar.
- Tené en cuenta el historial de la conversación para mantener coherencia con respuestas anteriores.
- Estructurá la respuesta en párrafos coherentes.

Contexto:
{context}

{chat_history}

Pregunta: {question}

Respuesta:"""
