import time
from langchain_core.documents import Document

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:
    import fitz  # PyMuPDF < 1.24 (legacy name)



def rewrite_query(
    user_query: str,
    chat_history: list[dict],
    groq_client,
) -> str:
    """
    Rephrases a follow-up question into a self-contained search query
    using the last few turns of conversation history.

    This is critical for accurate FAISS retrieval: raw follow-up questions
    like "What about the next section?" or "Can you elaborate?" are
    meaningless to a vector search without context.

    Uses llama-3.1-8b-instant (fast, cheap) for this lightweight rewrite task.
    Falls back to the original query on any error so the pipeline is never broken.

    Args:
        user_query:   The raw user question.
        chat_history: Full chat history list (dicts with 'role' and 'content').
        groq_client:  An initialised Groq client instance.

    Returns:
        A standalone, context-enriched search query string.
    """
    # Only rewrite if there is prior conversation context
    if not chat_history:
        return user_query

    # Build a compact summary of the last 3 exchanges (6 messages max)
    recent_turns = chat_history[-6:]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in recent_turns
    )

    rewrite_prompt = (
        "Given the following conversation history and a follow-up question, "
        "rewrite the follow-up question as a standalone, self-contained search query "
        "that captures all necessary context. "
        "Output ONLY the rewritten query — no explanation, no prefix, no quotes.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"Follow-up Question: {user_query}\n\n"
        "Standalone Search Query:"
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   # Fast, cheap model for this lightweight task
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        rewritten = response.choices[0].message.content.strip()
        # Safety: if the rewrite is empty or suspiciously short, fall back
        return rewritten if len(rewritten) > 5 else user_query
    except Exception:
        # Never break the pipeline — silently fall back to original query
        return user_query


def extract_text_from_pdf_bytes(file_bytes: bytes, filename: str) -> list[Document]:
    """
    Parses PDF bytes directly inside RAM without disk I/O lag.
    Returns a list of LangChain Documents, one per page.
    """
    documents = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    )
                )
    return documents


def is_rate_limited(
    user_id: str,
    rate_limit_dict: dict,
    max_requests: int = 10,
    window_seconds: int = 300,
    min_cooldown: int = 5
) -> tuple[bool, str | None]:
    """
    Enforces a rolling window rate-limit for anonymous sessions.

    Args:
        user_id:        Unique string identifying the user's browser tab session.
        rate_limit_dict: Shared mutable dict mapping user_id -> list of timestamps.
        max_requests:   Max queries allowed inside the window.
        window_seconds: Length of the rolling window in seconds (default 5 min).
        min_cooldown:   Minimum seconds between consecutive queries.

    Returns:
        (is_limited: bool, reason_message: str | None)
    """
    current_time = time.time()

    # First-time visitor — initialise their slot
    if user_id not in rate_limit_dict:
        rate_limit_dict[user_id] = []
        return False, None

    user_timestamps = rate_limit_dict[user_id]

    # 1. Per-query cooldown check
    if user_timestamps:
        time_since_last = current_time - user_timestamps[-1]
        if time_since_last < min_cooldown:
            wait = int(min_cooldown - time_since_last) + 1
            return True, f"Please wait {wait}s before your next query."

    # 2. Prune expired timestamps outside the rolling window
    rate_limit_dict[user_id] = [
        t for t in user_timestamps if current_time - t < window_seconds
    ]

    # 3. Window capacity check
    if len(rate_limit_dict[user_id]) >= max_requests:
        # BUG FIX: was `oldest_t = rate_limit_dict[user_id]` (assigned list)
        oldest_t = rate_limit_dict[user_id][0]
        reset_in = int(window_seconds - (current_time - oldest_t))
        return True, f"Rate limit reached (10 queries / 5 min). Resets in {reset_in}s."

    return False, None
