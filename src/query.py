from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.repository import get_retriever
from src.config import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, DEFAULT_SUBJECT_PROMPT
from src.token_tracker import format_chat_history

PHILOSOPHY_PROMPT = """Respondé la pregunta con rigor académico y precisión conceptual.

Directrices:
- Usá terminología precisa y específica de cada autor, corriente o época.
- Contextualizá histórica y filosóficamente los conceptos.
- Diferenciá las posturas de distintos autores cuando corresponda.
- No uses la oposición subjetivo/objetivo para explicar conceptos de filosofía clásica. Usá en su lugar los pares: opinable/verdadero, contingente/necesario, sensible/inteligible.
- Evitá estrictamente términos como "evidencia objetiva", "verdad absoluta", "demostrable", "prueba", "método científico", "conocimiento científico moderno" y "conocimiento científico" como traducción directa de episteme. Traducí episteme como "conocimiento fundamentado" o "saber necesario", sin asociarlo al método experimental moderno.
- Usá las categorías conceptuales propias de la filosofía clásica: mundo sensible/mundo inteligible, apariencia/esencia, universal/particular, necesario/contingente, mutable/inmutable, perfecto/corruptible.
- Estructurá la respuesta como una oposición binaria entre episteme y doxa, mostrando cómo la diferencia en el modo de conocer depende de la diferencia en el grado de ser del objeto tratado.
- Presentá la doxa no solo como ausencia de saber, sino como el grado de conocimiento que corresponde a lo que nace y perece (lo corruptible).
- Enmarcá la distinción en el contexto del tránsito del mito al logos, pero evitá simplificaciones lineales del tipo "mito = falsedad" y "logos = verdad científica". No asocies el logos con la racionalidad ilustrada o moderna; presentalo como la estructura racional e inteligible del cosmos que el filósofo intenta comprender.
- Usá ejemplos coherentes con el horizonte histórico del autor o período tratado. Si mencionás el mundo sensible, usá ejemplos de la naturaleza o la pólis (estrellas, leyes, objetos físicos), evitando tecnología o conceptos post-industriales.
- Evitá introducciones generales o definiciones enciclopédicas que no aporten directamente a la pregunta.
- Tené en cuenta el historial de la conversación para mantener coherencia con respuestas anteriores.
- Basate prioritariamente en el contexto proporcionado. Solo complementá con conocimiento general cuando sea necesario y sin introducir contradicciones conceptuales.

Contexto:
{context}

{chat_history}

Pregunta: {question}

Respuesta académica:"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _get_retriever(k: int = 7, materia: str = None):
    return get_retriever(k=k, materia=materia)


def _clean_response(text: str) -> str:
    import re
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE)


def _inject_history(template: str, history_text: str) -> str:
    if not history_text:
        return template.replace("{chat_history}", "")
    if "{chat_history}" in template:
        return template.replace("{chat_history}", history_text)
    return template.replace(
        "{question}",
        f"Historial de la conversación:\n{history_text}\n\nPregunta: {{question}}"
    )


def ask(question: str, k: int = 7, model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, materia: str = None, prompt_template: str = None, chat_history: list = None):
    retriever = _get_retriever(k=k, materia=materia)
    docs = retriever.invoke(question)

    template = prompt_template or DEFAULT_SUBJECT_PROMPT

    if chat_history:
        history_text = format_chat_history(chat_history)
        template = _inject_history(template, history_text)
    else:
        template = template.replace("{chat_history}", "")

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model=model, temperature=temperature)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = _clean_response(chain.invoke(question))

    sources = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Desconocido"),
            "page": doc.metadata.get("page", None),
        }
        for doc in docs
    ]

    return answer, sources
