# RAG PDF Chat

Sistema de preguntas y respuestas sobre PDFs usando RAG (Retrieval-Augmented Generation). Procesa documentos PDF, los indexa en una base vectorial y permite chatear con ellos organizados por **materias**.

## Stack

| Componente | Tecnología |
|---|---|
| Frontend | Streamlit |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, gratuito) |
| Vector store | ChromaDB (persistente) |
| LLM | Groq API (`llama-3.3-70b-versatile`, 128K context) |
| PDF parsing | PyMuPDF |
| Framework | LangChain |

## Funcionalidades

- **Materias** — Creá, editá y eliminá materias con prompts personalizados por tema
- **Subida de PDFs** — Asignale PDFs a una materia; se indexan automáticamente
- **Filtro por materia** — Consultá solo documentos de una materia específica o todos
- **Chat persistente** — El historial de la conversación se mantiene durante la sesión y se inyecta como contexto para respuestas coherentes
- **Barra de contexto** — Indicador visual del uso del contexto (128K tokens) con colores verde/naranja/rojo
- **Gestión de documentos** — Reasigná o eliminá documentos indexados desde la UI

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
2. Creá una materia desde el panel lateral
3. Subí PDFs asignados a esa materia
4. Seleccioná una materia y hacé preguntas sobre los documentos

Los PDFs quedan indexados en `chroma_db/` y persisten entre reinicios.

## Estructura

```
rag-pdf/
├── app.py                  # UI con Streamlit (sidebar, chat, barra de contexto)
├── requirements.txt        # Dependencias
├── .env.example            # Template de configuración
├── subjects.json           # Materias y prompts personalizados
├── file_index.json         # Índice de archivos indexados
├── chroma_db/              # Base vectorial persistente
└── src/
    ├── __init__.py
    ├── config.py           # Constantes (modelo, chunking, prompts por defecto)
    ├── embeddings.py       # Singleton de embeddings cacheados
    ├── ingestion.py        # Indexación de PDFs (texto, chunking, embeddings)
    ├── query.py            # Retrieval + consulta a Groq + inyección de historial
    ├── repository.py       # Wrapper de ChromaDB (alta/baja/consulta)
    ├── subjects.py         # CRUD de materias (JSON)
    ├── file_index.py       # CRUD de índice de archivos (JSON)
    └── token_tracker.py    # Estimación de tokens y cómputo de uso de contexto
```
