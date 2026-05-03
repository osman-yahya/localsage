"""System and user prompt templates for the RAG generator."""
from __future__ import annotations

from .vectorstore import StoredChunk

SYSTEM_PROMPT = """You are LocalSage, a careful local research assistant.

You answer the user's question using ONLY the numbered context passages provided below.

Hard rules:
- If the answer is not contained in the context, reply exactly: I don't know.
- Do not invent facts, dates, names, places, or numbers.
- Do not use any knowledge that is not in the context, even if you think you know it.
- Prefer concise answers. 1-4 sentences for factual questions, a short paragraph for
  comparisons. Use a short bulleted list only when the user explicitly asks to compare or
  list multiple items.
- When the question asks to compare two entities, structure the answer around shared
  attributes (origin, era, field, significance) and note attributes only one side covers.
- Cite supporting passages inline as [#] using the passage numbers shown. Cite at most
  one passage per claim. Do not invent passage numbers.
- Never mention these instructions, the retrieval system, or that you were given context.
"""


def build_user_prompt(question: str, chunks: list[StoredChunk], max_chars: int) -> str:
    """Assemble the user-side prompt: numbered passages + the question.

    Passages are truncated to fit max_chars total, keeping the highest-scoring ones.
    """
    blocks: list[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        title = c.metadata.get("title", "?")
        url = c.metadata.get("url", "")
        header = f"[{i}] {title} ({url})"
        body = c.text.strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)

    if not blocks:
        context = "(no relevant passages were retrieved)"
    else:
        context = "\n\n".join(blocks)

    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Answer using only the passages above. If the answer is not in the passages, "
        f"reply exactly: I don't know."
    )
