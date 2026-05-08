# RAG PDF Chat

Sistema de preguntas y respuestas sobre PDFs usando RAG (Retrieval-Augmented Generation). Procesa documentos PDF, los indexa en una base vectorial y permite chatear con ellos.

## Stack

| Componente | Tecnología |
|---|---|
| Frontend | Streamlit |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, gratuito) |
| Vector store | ChromaDB (persistente) |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| PDF parsing | PyMuPDF |
| Framework | LangChain |

## Requisitos

- Python 3.10+
- API key de [Groq](https://console.groq.com) (plan gratuito)

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows)
.venv\Scripts\Activate.ps1

# Activar (Linux/Mac)
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración

1. Copiá `.env.example` a `.env`
2. Agregá tu API key de Groq:

```
GROQ_API_KEY=gsk_tu-api-key
```

## Uso

```bash
streamlit run app.py
```

1. Ingresá tu API key de Groq en la interfaz (si no está en `.env`)
2. Subí uno o más PDFs
3. Hacé preguntas sobre los documentos

Los PDFs quedan indexados en `chroma_db/` y persisten entre reinicios.

## Estructura

```
rag-pdf/
├── app.py              # UI con Streamlit
├── ingestion.py        # Indexación de PDFs (texto, chunking, embeddings, ChromaDB)
├── query.py            # Retrieval + consulta a Groq
├── embeddings.py       # Singleton de embeddings cacheados
├── requirements.txt    # Dependencias
├── .env.example        # Template de configuración
├── .gitignore
└── README.md
```
