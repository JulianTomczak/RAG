from src.config import MODEL_MAX_TOKENS


def estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


def format_chat_history(messages, max_turns=3):
    lines = []
    for msg in messages[-(max_turns * 2):]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        content = msg.get("content", "")[:500]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def compute_usage(messages, question, sources_text, template):
    hist_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    question_tokens = estimate_tokens(question)
    sources_tokens = estimate_tokens(sources_text)
    template_tokens = estimate_tokens(template)
    total = hist_tokens + question_tokens + sources_tokens + template_tokens
    fraction = min(total / MODEL_MAX_TOKENS, 1.0)
    return int(total), round(fraction, 3)
