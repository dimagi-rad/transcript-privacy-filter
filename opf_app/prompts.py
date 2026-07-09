from __future__ import annotations


REDACTION_PROMPT_VERSION = "v2-initial-2026-07-09"

REDACTION_PROMPT = """You redact transcript sentences for privacy.

Return only the structured output requested by the schema.

For each input sentence, produce a redacted version that preserves meaning, grammar, punctuation, language, and non-sensitive context.

Replace private identifying information with typed placeholders such as <PRIVATE_PERSON>, <PRIVATE_EMAIL>, <PRIVATE_PHONE>, <PRIVATE_ADDRESS>, <PRIVATE_DATE>, <PRIVATE_URL>, <ACCOUNT_NUMBER>, or <SECRET>.

Redact private people, private contact details, private addresses, private account or identifier numbers, private URLs, secrets, and dates that identify a private person or private event.

Do not redact public figures, public institutions, public place names, general clinical content, generic demographic terms, chatbot names, study concepts, or ordinary non-identifying words.

Do not add explanations.
Do not change protected mask tokens that look like __KEEP_000001__.
Do not add, remove, merge, split, reorder, or summarize sentences.
"""


def build_redaction_prompt() -> str:
    """Return the reviewed v2 sentence-redaction prompt asset."""
    return REDACTION_PROMPT.strip()
