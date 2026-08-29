import re

ROLE_PREFIX_PATTERN = re.compile(r"(?im)^\s*(system|assistant|user)\s*:\s*")


def sanitize_question(question: str, *, max_length: int = 500) -> str:
    sanitized = ROLE_PREFIX_PATTERN.sub("", question)

    sanitized = sanitized.strip()

    return sanitized[:max_length]
