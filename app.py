import os
import time

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_navigation_bar import st_navbar

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

from utils import extract_text_from_pdf_bytes, is_rate_limited, rewrite_query

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Ragobot – RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, body, .stApp {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Background ── */
.stApp { background: #0d0e1a !important; color: #e0e6f0 !important; }

/* ── Navbar ── */
.stNavigationBar {
    background: #13142a !important;
    border-radius: 14px;
    box-shadow: 0 8px 32px rgba(30,40,80,0.35);
    margin-top: 18px !important;
    margin-bottom: 32px !important;
    padding: 0.5rem 0;
    border-bottom: 2px solid #4f7cff;
}
.stNavigationBar span {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    border-radius: 10px !important;
    margin: 0 6px !important;
    transition: background 0.2s, color 0.2s;
    color: #b0bacc !important;
}
.stNavigationBar span.active {
    background: #4f7cff !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(79,124,255,0.35);
}
.stNavigationBar span:hover { background: #23253d !important; color: #fff !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13142a !important;
    border-right: 1px solid #222440;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Heading box ── */
.heading-box {
    background: linear-gradient(135deg, #1a1d3a 0%, #23274a 100%);
    border-left: 4px solid #4f7cff;
    color: #fff;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.5px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #1a1d3a !important;
    color: #e0e6f0 !important;
    border: 1px solid #2e3260 !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f7cff, #7b5ea7);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Metric cards ── */
.metric-card {
    background: #13142a;
    border: 1px solid #222440;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #4f7cff;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    color: #7888aa;
    margin-top: 4px;
}

/* ── Info / success banners ── */
.info-banner {
    background: #1a2740;
    border-left: 4px solid #4f7cff;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #c0ccdd;
}

/* ── Citation card ── */
.citation-card {
    background: #181a30;
    border: 1px solid #2e3260;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.citation-card .cite-header {
    font-weight: 700;
    color: #4f7cff;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.citation-card .cite-snippet {
    font-size: 0.82rem;
    color: #8899bb;
    font-style: italic;
    line-height: 1.5;
}

/* ── Footer ── */
.footer {
    position: fixed; left: 0; bottom: 0; width: 100%;
    background: #0d0e1a;
    border-top: 1px solid #1e2040;
    color: #4f7cff;
    text-align: center;
    padding: 10px 0;
    font-size: 0.82rem;
    z-index: 200;
}
.stApp { padding-bottom: 52px !important; }

/* ── Rate-limit badge ── */
.rate-badge {
    display: inline-block;
    background: #2a1a3a;
    border: 1px solid #7b5ea7;
    color: #c9a0ff;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------
selected_page = st_navbar([
    "Home", "How to Use", "About Us", "Contact Us", "Future Enhancements"
])

# ---------------------------------------------------------------------------
# Cached shared resources (load once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (one-time setup)…")
def load_shared_resources():
    """
    Initialises the nomic-embed-text-v1.5 embedding model and the global
    rate-limit tracker exactly once, shared across all browser sessions.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
    )
    global_rate_tracker: dict = {}
    return embeddings, global_rate_tracker


embeddings, global_rate_tracker = load_shared_resources()

# ---------------------------------------------------------------------------
# Per-tab session state initialisation
# ---------------------------------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ---------------------------------------------------------------------------
# Groq API key — loaded from environment / Streamlit Secrets only.
# Never displayed in the UI.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected_page == "Home":
    st.markdown("<div class='heading-box'>🔬 Ragobot — RAG Research Assistant</div>",
                unsafe_allow_html=True)

    # API key comes exclusively from environment / Streamlit Secrets — never from user input
    active_api_key = GROQ_API_KEY

    # Fixed chunking parameters — not exposed to users
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDFs (100+ pages supported)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are processed entirely in RAM and never written to disk.",
        )

        with st.form("ingestion_form"):
            index_btn = st.form_submit_button("⚡ Index Documents", type="primary")

        st.markdown("---")

        # Reset session button
        if st.button("🗑️ Reset Session & Clear Data", use_container_width=True):
            st.session_state.vector_db = None
            st.session_state.chat_history = []
            st.session_state.uploaded_filenames = []
            st.session_state.total_chunks = 0
            st.success("Session cleared! All vectors and chat history wiped.")
            st.rerun()

        # Contact info
        st.markdown("""
        <div style='background:#1a1d3a;border-radius:10px;padding:14px 16px;margin-top:12px;font-size:0.85rem;'>
            <b style='color:#4f7cff;'>Contact</b><br>
            📞 <a href='tel:7004918026' style='color:#c0ccdd;'>+91-7004918026</a><br>
            ✉️ <a href='mailto:as120171.omkumar@gmail.com' style='color:#c0ccdd;'>as120171.omkumar@gmail.com</a><br>
            💻 <a href='https://github.com/omsingh031' target='_blank' style='color:#c0ccdd;'>GitHub</a> ·
            🔗 <a href='https://linkedin.com/in/omsingh031' target='_blank' style='color:#c0ccdd;'>LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    if index_btn:
        if not active_api_key:
            st.error("⚠️ App configuration error: GROQ_API_KEY is not set. Please contact the administrator.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF before indexing.")
        else:
            with st.spinner("⚡ Parsing & embedding documents into FAISS…"):
                all_raw_docs = []
                filenames = []
                for file in uploaded_files:
                    pages = extract_text_from_pdf_bytes(file.read(), file.name)
                    all_raw_docs.extend(pages)
                    filenames.append(file.name)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,

                    separators=["\n\n", "\n", " ", ""],
                )
                chunks = splitter.split_documents(all_raw_docs)

                if chunks:
                    st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
                    st.session_state.uploaded_filenames = filenames
                    st.session_state.total_chunks = len(chunks)
                    # Reset chat but keep the index
                    st.session_state.chat_history = []
                    st.success(
                        f"✅ Indexed **{len(chunks):,}** chunks from "
                        f"**{len(filenames)}** file(s). Start asking questions below!"
                    )
                else:
                    st.error("No valid text could be extracted from the uploaded files.")

    # ── Status metrics ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(st.session_state.uploaded_filenames)}</div>
            <div class='metric-label'>Files Indexed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{st.session_state.total_chunks:,}</div>
            <div class='metric-label'>Vector Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        msg_count = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{msg_count}</div>
            <div class='metric-label'>Queries This Session</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Empty state ───────────────────────────────────────────────────────────
    if st.session_state.vector_db is None:
        st.markdown("""
        <div class='info-banner'>
            👈 <b>Upload PDFs</b> in the sidebar, then click <b>Index Documents</b> to begin.
            Your data lives only in your browser session — it vanishes automatically when you close the tab.
        </div>""", unsafe_allow_html=True)

    # ── Chat interface ────────────────────────────────────────────────────────
    else:
        # Source filter
        selected_papers = st.multiselect(
            "🗂️ Filter search scope by file:",
            options=st.session_state.uploaded_filenames,
            default=st.session_state.uploaded_filenames,
            help="Deselect files to exclude them from retrieval.",
        )

        st.markdown("---")

        # Render existing chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander("📚 View Source Citations"):
                        for idx, src in enumerate(msg["citations"]):
                            st.markdown(f"""
                            <div class='citation-card'>
                                <div class='cite-header'>[{idx+1}] {src['file']} — Page {src['page']}</div>
                                <div class='cite-snippet'>"{src['text'][:200]}…"</div>
                            </div>""", unsafe_allow_html=True)

        # Chat input
        if user_query := st.chat_input("Ask a question about your documents…"):
            # ── Rate limiting ─────────────────────────────────────────
            ctx = get_script_run_ctx()
            user_session_id = ctx.session_id if ctx else "global_guest"

            limited, limit_msg = is_rate_limited(user_session_id, global_rate_tracker)
            if limited:
                st.error(f"🛑 {limit_msg}")
            else:
                # Log timestamp
                if user_session_id not in global_rate_tracker:
                    global_rate_tracker[user_session_id] = []
                global_rate_tracker[user_session_id].append(time.time())

                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_query)
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_query}
                )

                # Initialise Groq client once — reused for rewriting + LLM answer
                groq_client = Groq(api_key=active_api_key)

                # ── Query Rewriting ───────────────────────────────────
                # Rephrase follow-up questions into standalone FAISS queries
                # using the conversation history for context.
                prior_history = st.session_state.chat_history[:-1]
                with st.spinner("🔍 Analysing question context…"):
                    search_query = rewrite_query(user_query, prior_history, groq_client)

                # ── Retrieval ─────────────────────────────────────────
                raw_matches = st.session_state.vector_db.similarity_search(
                    search_query, k=5
                )
                filtered = [
                    d for d in raw_matches
                    if d.metadata.get("source") in selected_papers
                ]
                if not filtered:
                    filtered = raw_matches  # fallback if no filter active

                # Build context blocks and citation list
                context_blocks, citations = [], []
                for doc in filtered:
                    src = doc.metadata.get("source", "Unknown")
                    pg = doc.metadata.get("page", "?")
                    context_blocks.append(
                        f"[Source: {src}, Page {pg}]:\n{doc.page_content}"
                    )
                    citations.append({"file": src, "page": pg, "text": doc.page_content})

                compiled_context = "\n\n---\n\n".join(context_blocks)

                # Build multi-turn message history for Groq
                groq_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional research assistant. "
                            "Answer rigorously using the provided document context. "
                            "Always cite the source filename and page number in your answer. "
                            "If the answer is not in the context, say so clearly."
                        ),
                    }
                ]
                # Inject prior turns (last 12 messages max for token budget)
                for past in st.session_state.chat_history[-12:]:
                    groq_messages.append({"role": past["role"], "content": past["content"]})
                # Replace last user message with context-enriched version
                groq_messages[-1]["content"] = (
                    f"Document Context:\n{compiled_context}\n\nQuestion: {user_query}"
                )

                # ── Groq call with retry ──────────────────────────────
                with st.chat_message("assistant"):
                    with st.spinner("Thinking via Groq llama-3.3-70b-versatile…"):
                        ai_response = None
                        for attempt in range(2):
                            try:
                                completion = groq_client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=groq_messages,
                                    temperature=0.1,
                                    max_tokens=1024,
                                )
                                ai_response = completion.choices[0].message.content
                                break
                            except Exception as exc:
                                if "rate_limit" in str(exc).lower() and attempt == 0:
                                    time.sleep(2)
                                    continue
                                st.error(f"Groq error: {exc}")
                                break

                    if ai_response:
                        st.markdown(ai_response)
                        with st.expander("📚 View Source Citations"):
                            for idx, src in enumerate(citations):
                                st.markdown(f"""
                                <div class='citation-card'>
                                    <div class='cite-header'>[{idx+1}] {src['file']} — Page {src['page']}</div>
                                    <div class='cite-snippet'>"{src['text'][:200]}…"</div>
                                </div>""", unsafe_allow_html=True)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": ai_response,
                            "citations": citations,
                        })




# ══════════════════════════════════════════════════════════════════════════════
# HOW TO USE
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "How to Use":
    st.markdown("<div class='heading-box'>📖 How to Use</div>", unsafe_allow_html=True)
    st.markdown("""
### What This App Does

**Ragobot** is a Retrieval-Augmented Generation (RAG) system that lets you interrogate
long academic PDFs (100+ pages) with a conversational AI, while keeping your data
completely private and session-isolated.

---

### 🚀 Step-by-Step Guide

| Step | Action |
|------|--------|
| **1** | Click **Browse Files** in the sidebar and select one or more PDFs |
| **2** | Click **⚡ Index Documents** — the app parses, chunks & embeds your PDFs into FAISS |
| **3** | Use the **file filter** multi-select to narrow retrieval to specific papers |
| **4** | Type your question in the chat bar and hit Enter |
| **5** | View the AI response and expand **📚 View Source Citations** to trace each answer |
| **6** | Click **🗑️ Reset Session** to wipe all data and start fresh |

---

### 🔒 Privacy Architecture

- Your PDFs are **never written to disk** — all processing is in RAM.
- The FAISS index lives **only** in your browser tab's session state.
- Closing the tab triggers Python garbage collection — vectors are permanently deleted.
- **No cross-user data leakage** is possible by design.

---

### ⚡ Rate Limits

To protect the shared Groq API key, each session is capped at **10 queries per 5 minutes**
with a minimum 5-second cooldown between queries.
""")


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT US
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "About Us":
    st.markdown("<div class='heading-box'>🤖 About Ragobot</div>", unsafe_allow_html=True)
    st.markdown("""
Welcome to **Ragobot** — an intelligent research assistant that bridges human curiosity
and machine knowledge through cutting-edge AI.

---

### 🎯 Our Mission

To make AI more human-centric by combining advanced language models with intuitive user
interfaces and real-world usability — enabling anyone to have a conversation with their
research documents.

---

### 🛠️ Technical Architecture

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq `llama-3.3-70b-versatile` |
| **Embeddings** | `nomic-ai/nomic-embed-text-v1.5` (8K context, open-source) |
| **Vector Store** | FAISS (in-memory, CPU) |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **PDF Parsing** | PyMuPDF (in-RAM, no disk I/O) |
| **UI** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

### 💼 What We Do

- Build LLM-powered apps with real-world utility
- Craft sleek UI experiences using Streamlit
- Apply retrieval techniques like vector search with LangChain and FAISS
- Continuously explore the boundaries of AI, UX, and automation
""")




# ══════════════════════════════════════════════════════════════════════════════
# CONTACT US
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "Contact Us":
    st.markdown("<div class='heading-box'>📫 Contact Us</div>", unsafe_allow_html=True)
    st.markdown("""
| Channel | Details |
|---------|---------|
| 📞 Phone | [+91-7004918026](tel:7004918026) |
| ✉️ Email | [as120171.omkumar@gmail.com](mailto:as120171.omkumar@gmail.com) |
| 📷 Instagram | [@omsingh031](https://www.instagram.com/omsingh031/) |
| 💻 GitHub | [omsingh031](https://github.com/omsingh031) |
| 🔗 LinkedIn | [omsingh031](https://linkedin.com/in/omsingh031) |
""")


# ══════════════════════════════════════════════════════════════════════════════
# FUTURE ENHANCEMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "Future Enhancements":
    st.markdown("<div class='heading-box'>🛣️ Future Enhancements</div>", unsafe_allow_html=True)
    st.markdown("""
| Enhancement | Description |
|-------------|-------------|
| 🌐 **Web URL Ingestion** | Scrape and index web articles alongside PDFs |
| 📊 **Multi-modal Support** | Parse tables, charts, and images from PDFs |
| 🔐 **Authentication** | User accounts with persistent named sessions |
| 📤 **Export Chat Logs** | Download full conversation + citations as PDF/Markdown |
| 🧠 **Hybrid Search** | Combine FAISS dense search with BM25 sparse search for higher recall |
| 🗃️ **Multi-collection** | Separate vector stores per project / research topic |
| 📈 **Analytics Dashboard** | Usage stats, query trends, most-cited pages |
""")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    © 2026 Om Kumar Singh · Ragobot – RAG Research Assistant · All rights reserved.
</div>
""", unsafe_allow_html=True)
