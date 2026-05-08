import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from ingestion import index_pdf, list_indexed_files, delete_file_from_index
from query import ask

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = bool(os.getenv("GROQ_API_KEY"))

def configure_api_key():
    key = st.session_state.api_key_input
    if key:
        os.environ["GROQ_API_KEY"] = key
        st.session_state.api_key_configured = True
        st.rerun()


with st.sidebar:
    st.title("📄 RAG PDF Chat")

    if not st.session_state.api_key_configured:
        st.warning("Configurá tu API key de Groq")
        st.text_input(
            "Groq API Key",
            type="password",
            key="api_key_input",
            on_change=configure_api_key,
        )
        st.stop()

    st.divider()
    st.subheader("Subir PDFs")
    uploaded_files = st.file_uploader(
        "Arrastrá PDFs acá",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Procesando PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                chunks = index_pdf(tmp_path, uploaded_file.name)
                os.unlink(tmp_path)

                if chunks > 0:
                    st.success(f"✅ {uploaded_file.name} — {chunks} chunks indexados")
                else:
                    st.info(f"⏭️ {uploaded_file.name} ya estaba indexado")

    st.divider()
    st.subheader("Documentos indexados")
    indexed = list_indexed_files()
    if indexed:
        for doc in indexed:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {doc}")
            with col2:
                if st.button("🗑️", key=f"del_{doc}"):
                    delete_file_from_index(doc)
                    st.rerun()
    else:
        st.caption("No hay documentos indexados")

    st.divider()
    if st.button("🧹 Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()


st.title("💬 Chat con tus PDFs")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Fuentes"):
                for i, src in enumerate(msg["sources"], 1):
                    page_info = f" (pág. {src['page']})" if src["page"] is not None else ""
                    st.markdown(f"**Fuente {i}:** `{src['source']}`{page_info}")
                    st.text(src["content"][:500] + ("..." if len(src["content"]) > 500 else ""))

if prompt := st.chat_input("Hacé una pregunta sobre los PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos..."):
            try:
                answer, sources = ask(prompt)
                st.markdown(answer)

                if sources:
                    with st.expander("📚 Fuentes"):
                        for i, src in enumerate(sources, 1):
                            page_info = f" (pág. {src['page']})" if src["page"] is not None else ""
                            st.markdown(f"**Fuente {i}:** `{src['source']}`{page_info}")
                            st.text(src["content"][:500] + ("..." if len(src["content"]) > 500 else ""))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
