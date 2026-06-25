

def build_context(history: list, limit: int = 10) -> str:
    """
    Convert conversation history into prompt-friendly text.
    """

    return "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in history[-limit:]
    )
