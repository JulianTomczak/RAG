import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import index_pdf
from src import subjects
from src import file_index
from src import repository
from src.query import ask
from src.config import DEFAULT_SUBJECT_PROMPT, MODEL_MAX_TOKENS
from src.token_tracker import estimate_tokens, compute_usage

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = bool(os.getenv("GROQ_API_KEY"))
if "edit_subject" not in st.session_state:
    st.session_state.edit_subject = None
if "edit_file_materia" not in st.session_state:
    st.session_state.edit_file_materia = None

# Custom CSS
st.markdown("""
<style>
div[data-testid="stChatInput"] textarea:focus,
div[data-testid="stChatInput"] textarea:active {
    border-color: #999 !important;
    box-shadow: 0 0 0 1px #999 !important;
}
div[data-testid="stChatInput"] textarea {
    border-color: #ccc !important;
}
div[data-testid="stChatInput"] button {
    color: #666 !important;
}
div.stChatFloatingInputContainer {
    border-top: 1px solid #ddd !important;
}
.section-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #555;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
div.stProgress > div > div > div > div {
    background-color: #888;
}
</style>
""", unsafe_allow_html=True)


def configure_api_key():
    key = st.session_state.api_key_input
    if key:
        os.environ["GROQ_API_KEY"] = key
        st.session_state.api_key_configured = True
        st.rerun()


# --- Sidebar ---
with st.sidebar:
    st.title("📄 RAG PDF Chat")

    if not st.session_state.api_key_configured:
        st.warning("Configurá tu API key de Groq")
        st.text_input("Groq API Key", type="password", key="api_key_input", on_change=configure_api_key)
        st.stop()

    # --- Subject management ---
    st.divider()
    st.markdown('<p class="section-header">Materias</p>', unsafe_allow_html=True)

    subject_names = subjects.list_subjects()

    # Create new subject
    with st.expander("➕ Nueva materia", expanded=False):
        new_name = st.text_input("Nombre", key="new_subj_name", placeholder="Ej: Epistemología")
        new_prompt = st.text_area(
            "Prompt inicial",
            value=DEFAULT_SUBJECT_PROMPT,
            height=200,
            key="new_subj_prompt",
            help="El prompt le dice al modelo CÓMO responder. Debe incluir {{context}} y {{question}}.",
        )
        if st.button("Crear materia", use_container_width=True, type="primary"):
            if new_name.strip():
                if subjects.create_subject(new_name.strip(), new_prompt.strip()):
                    st.success(f"'{new_name.strip()}' creada")
                    st.rerun()
                else:
                    st.warning("Esa materia ya existe")

    # Subject list with edit/delete
    if subject_names:
        for s in subject_names:
            cols = st.columns([3, 1, 1])
            cols[0].write(f"📁 {s}")
            if cols[1].button("✏️", key=f"edit_{s}", help="Editar prompt"):
                st.session_state.edit_subject = s
                st.rerun()
            if cols[2].button("🗑️", key=f"del_{s}", help="Eliminar materia y sus PDFs"):
                subjects.delete_subject(s)
                st.rerun()

        # Prompt editor for selected subject
        if st.session_state.edit_subject and st.session_state.edit_subject in subject_names:
            current_prompt = subjects.get_subject_prompt(st.session_state.edit_subject) or DEFAULT_SUBJECT_PROMPT
            with st.container(border=True):
                st.caption(f"Editando prompt: **{st.session_state.edit_subject}**")
                edited = st.text_area(
                    "Prompt",
                    value=current_prompt,
                    height=250,
                    key="prompt_editor",
                    help="Debe incluir {{context}} y {{question}}.",
                )
                c1, c2 = st.columns(2)
                if c1.button("💾 Guardar", use_container_width=True, type="primary"):
                    subjects.update_subject_prompt(st.session_state.edit_subject, edited)
                    st.session_state.edit_subject = None
                    st.rerun()
                if c2.button("Cancelar", use_container_width=True):
                    st.session_state.edit_subject = None
                    st.rerun()
    else:
        st.caption("Sin materias todavía")

    # --- Subject filter for chat ---
    st.divider()
    subject_options = ["Todas"] + subject_names
    selected_subject = st.selectbox("Consultar en:", subject_options, key="subject_filter")
    materia_filter = selected_subject if selected_subject != "Todas" else None

    if materia_filter:
        current_prompt = subjects.get_subject_prompt(materia_filter) or DEFAULT_SUBJECT_PROMPT
        preview = current_prompt[:120].replace("\n", " ")
        st.caption(f"Prompt: _{preview}..._")

    # --- PDF Upload ---
    st.divider()
    st.markdown('<p class="section-header">Subir PDFs</p>', unsafe_allow_html=True)

    if subject_names:
        upload_subject = st.selectbox("Asignar a:", subject_names, key="upload_subject", label_visibility="collapsed")
    else:
        st.warning("Creá una materia primero")
        upload_subject = None

    uploaded_files = st.file_uploader(
        "Seleccioná PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and upload_subject:
        with st.spinner("Procesando PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                chunks = index_pdf(tmp_path, uploaded_file.name, materia=upload_subject)
                os.unlink(tmp_path)

                if chunks > 0:
                    st.success(f"✅ {uploaded_file.name} — {chunks} chunks")
                else:
                    st.info(f"⏭️ {uploaded_file.name} ya estaba indexado")

    # --- Indexed Documents ---
    st.divider()
    with st.expander("📚 Documentos indexados", expanded=False):
        indexed = file_index.list_files()
        if indexed:
            file_map = {d["filename"]: d["materia"] for d in indexed}
            by_subject = {}
            for doc in indexed:
                mat = doc["materia"] or "Sin materia"
                by_subject.setdefault(mat, []).append(doc["filename"])

            for mat in sorted(by_subject.keys()):
                with st.container(border=True):
                    st.caption(f"📁 {mat} ({len(by_subject[mat])})")
                    for fname in by_subject[mat]:
                        c1, c2, c3 = st.columns([3, 1, 1])
                        c1.caption(f"📄 {fname}")
                        if c2.button("✏️", key=f"move_{fname}", help="Cambiar materia"):
                            st.session_state.edit_file_materia = fname
                            st.rerun()
                        if c3.button("🗑️", key=f"del_doc_{fname}", help=f"Eliminar {fname}"):
                            repository.delete_file(fname)
                            file_index.remove_file(fname)
                            st.rerun()

                    # Inline materia editor for the file being edited in this group
                    if st.session_state.edit_file_materia in by_subject[mat]:
                        ef = st.session_state.edit_file_materia
                        cur_mat = file_map.get(ef, "")
                        opts = subject_names + ["(Sin materia)"]
                        def_idx = subject_names.index(cur_mat) + 1 if cur_mat in subject_names else 0
                        new_mat = st.selectbox(
                            f"Asignar materia a `{ef}`",
                            opts,
                            index=def_idx,
                            key=f"mat_select_{ef}",
                        )
                        c1, c2 = st.columns(2)
                        if c1.button("💾 Guardar", key=f"save_mat_{ef}", use_container_width=True, type="primary"):
                            repository.update_file_metadata(ef, new_mat if new_mat != "(Sin materia)" else "")
                            file_index.update_materia(ef, new_mat if new_mat != "(Sin materia)" else "")
                            st.session_state.edit_file_materia = None
                            st.rerun()
                        if c2.button("Cancelar", key=f"cancel_mat_{ef}", use_container_width=True):
                            st.session_state.edit_file_materia = None
                            st.rerun()
        else:
            st.caption("No hay documentos indexados")

    # --- Context usage bar ---
    st.divider()
    if "last_usage" in st.session_state:
        total_tokens, fraction = st.session_state.last_usage
        color = "green" if fraction < 0.8 else ("orange" if fraction < 0.95 else "red")
        st.markdown(f'<p style="color:{color};font-size:0.85rem;font-weight:600;">🧠 Contexto usado</p>', unsafe_allow_html=True)
        st.progress(fraction)
        st.caption(f"~{total_tokens // 1000}K / {MODEL_MAX_TOKENS // 1000}K tokens")
        if fraction > 0.95:
            st.warning("⚠️ Contexto casi lleno. Limpiá la conversación.")

    # --- Clear chat ---
    st.divider()
    if st.button("🧹 Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_usage = None
        st.rerun()


# --- Main chat area ---
st.title("💬 Chat con tus PDFs")

if materia_filter:
    st.caption(f"Consultando solo en: **{materia_filter}**")

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
                subj_prompt = None
                if materia_filter:
                    subj_prompt = subjects.get_subject_prompt(materia_filter)

                template = subj_prompt or DEFAULT_SUBJECT_PROMPT
                previous = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else None

                answer, sources = ask(
                    prompt,
                    materia=materia_filter,
                    prompt_template=subj_prompt,
                    chat_history=previous,
                )
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

                context_text = " ".join(s["content"] for s in sources)
                total, frac = compute_usage(st.session_state.messages, prompt, context_text, template)
                st.session_state.last_usage = (total, frac)

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
