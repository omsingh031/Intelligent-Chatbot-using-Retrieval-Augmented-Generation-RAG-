import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
from dotenv import load_dotenv 
import os
import streamlit as st
from streamlit_community_navigation_bar import st_navbar
import warnings
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

warnings.filterwarnings('ignore')
load_dotenv()

# API environment
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "store" not in st.session_state:
    st.session_state["store"] = {}

CHROMA_DB_DIR = "chroma_db"
from langchain.vectorstores import Chroma
# For Streamlit Cloud deployments
CHROMA_DB_DIR = "/tmp/chroma_db"  # Temporary directory with write access


def create_vectorstore_from_docs(docs, embedding):
    return Chroma.from_documents(docs, embedding_function=embedding)


try:
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)
except Exception as e:
    st.error(f"Error initializing Gemini models: {e}")
    st.stop()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

st.set_page_config(page_title="RAG with Conversational Memory", layout="wide")
st.markdown("""
    <style>
        body, .stApp {
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif !important;
            color: #e0e6f0 !important;
        }
        .stNavigationBar {
            background: #23243a !important;
            border-radius: 14px;
            box-shadow: 0 8px 32px rgba(30,40,80,0.25), 0 1.5px 6px rgba(0,0,0,0.12);
            margin-top: 18px !important;
            margin-bottom: 36px !important;
            padding: 0.7rem 0;
            min-height: 60px;
            display: flex;
            align-items: center;
            z-index: 100;
            border-bottom: 2px solid #3e68ff;
        }
        .stNavigationBar span {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            padding: 12px 36px !important;
            border-radius: 10px !important;
            margin: 0 10px !important;
            transition: background 0.2s, color 0.2s;
            color: #e0e6f0 !important;
        }
        .stNavigationBar span.active {
            background: #3e68ff !important;
            color: #fff !important;
            box-shadow: 0 2px 8px rgba(62,104,255,0.18);
        }
        .stNavigationBar span:hover {
            background: #35365a !important;
            color: #fff !important;
        }
        .heading-box {
            background-color:#35365a;
            color:#fff;
            border-radius:16px;
            padding:18px 28px 14px 28px;
            margin-bottom:18px;
            font-size:2.1rem;
            font-weight:700;
            text-align:left;
            letter-spacing:1px;
        }
        .history-box {
            background-color:#35365a;
            color:#fff;
            border-radius:12px;
            padding:12px 22px 10px 22px;
            margin-top:24px;
            margin-bottom:10px;
            font-size:1.1rem;
            font-weight:500;
            text-align:left;
            letter-spacing:0.5px;
        }
        .history-box .history-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 12px;
            margin-top: 4px;
        }
        .main-content {
            padding: 0 32px 24px 32px;
        }
        .stTextInput>div>div>input { 
            background-color: #2d2d44; color: white; 
            border-radius: 8px; border: 1px solid #444;
        }
        .stButton>button {
            background-color: #3e68ff; color: white;
            border-radius: 8px; border: none;
            padding: 10px 20px;
        }
        .stButton>button:hover { background-color: #5e78ff; }
        .chat-history {
            background-color: #2d2d44; border: 1px solid #444; border-radius: 8px;
            padding: 15px; margin-top: 10px; max-height: 400px; overflow-y: auto;
        }
        [data-testid="stSidebar"] {
            background-color: #1e1e2e;
            border-right: 1px solid #383850;
        }
        .contact-box {
            background-color: #ccccc6  ;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #040760 ;
        }
        .contact-title {
            font-size: 20px; font-weight: bold; color: #1a73e8; margin-bottom: 10px;
        }
        .contact-item { margin-bottom: 8px; }
        .contact-icon { margin-right: 8px; }
        a.contact-link { color: #1a73e8; text-decoration: none; }
        a.contact-link:hover { text-decoration: underline; }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #1e1e2e;
            border-top: 1px solid #383850;
            color:#FFC300 ;
            text-align: center;
            padding: 10px 0;
            z-index: 100;
        }
        .stApp {
            padding-bottom: 50px !important;
        }
    </style>
""", unsafe_allow_html=True)

selected_page = st_navbar(
    ["Home", "How to Use", "About Us", "Team", "Contact Us", "Future Enhancements"]
)

# --- MAIN CONTENT ---
if selected_page == "Home":
    st.markdown("<div class='heading-box'>Intelligent Chatbot using Retrieval Augmented Generation(RAG)</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("""
### 📌 **Welcome to the RAG Chatbot with Memory!**

Harness the power of ****AI + Retrieval**** to get precise, document-specific answers - whether you're researching, studying, or building intelligent systems.

---
    """)
    # Chat interface and sidebar
    with st.sidebar:
        st.header("Data Source")
        source_option = st.radio("Select a source:", ("Web URL", "Upload File"))
        url, uploaded_file = None, None
        load_url_button = load_file_button = False

        if source_option == "Web URL":
            url = st.text_input("Enter the URL to scrape:")
            load_url_button = st.button("Load URL")
        elif source_option == "Upload File":
            uploaded_file = st.file_uploader("Upload a text file or PDF:", type=["txt", "pdf"])
            load_file_button = st.button("Load File")
            
        st.markdown("""
            <div class='contact-box'>
                <div class='contact-title'>Contact Us</div>
                <div class='contact-item'><span class='contact-icon'>📞</span><a class='contact-link' href='tel:7004918026'>+91-7004918026</a></div>
                <div class='contact-item'><span class='contact-icon'>✉️</span><a class='contact-link' href='mailto:as120171.omkumar@gmail.com'>as120171.omkumar@gmail.com</a></div>
                <div class='contact-item'><span class='contact-icon'>📷</span><a class='contact-link' href='https://www.instagram.com/omsingh031/' target='_blank'>Instagram</a></div>
                <div class='contact-item'><span class='contact-icon'>💻</span><a class='contact-link' href='https://github.com/omsingh031' target='_blank'>GitHub</a></div>
                <div class='contact-item'><span class='contact-icon'>🔗</span><a class='contact-link' href='https://linkedin.com/in/omsingh031' target='_blank'>LinkedIn</a></div>
            </div>
        """, unsafe_allow_html=True)

    session_id = "default_session"
    chat_history = st.session_state["store"].get(session_id, ChatMessageHistory()).messages if "store" in st.session_state else []

    try:
        vectorstore = st.session_state.get('vectorstore') or Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=gemini_embeddings)
        st.session_state['vectorstore'] = vectorstore
    except Exception as e:
        st.error(f"Error initializing or loading Chroma DB: {e}")
        st.stop()

    if (source_option == "Web URL" and url and load_url_button) or (
        source_option == "Upload File" and uploaded_file and load_file_button):
        try:
            if source_option == "Web URL":
                loader = WebBaseLoader(web_paths=(url,))
                doc = loader.load()
                text_content = "".join([element.page_content + "\n" for element in doc])
                class DummyDocument:
                    def __init__(self, page_content, metadata=None):
                        self.page_content = page_content
                        self.metadata = metadata or {}
                doc = [DummyDocument(text_content, metadata={"source": url})]
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                loader = TextLoader(temp_file_path) if "text" in uploaded_file.type else PyPDFLoader(temp_file_path)
                doc = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(doc)
            if not splits:
                st.error("No content extracted. Check your input.")
                st.stop()

            st.session_state['vectorstore'].add_documents(documents=splits)
            retriever = st.session_state['vectorstore'].as_retriever()

            system_prompt = ("You are an assistant for question-answering tasks. Use the retrieved context."
                             "If you don't know the answer, say that you don't. Keep answers concise.\n\n{context}")

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given chat history and latest user question, rephrase it standalone."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            rag_chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(model, qa_prompt))
            st.session_state["rag_chain"] = rag_chain
            st.success("Document loaded successfully! Ask a question below.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    user_question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if "rag_chain" not in st.session_state:
            st.warning("Please load a document or URL first.")
        else:
            try:
                conversational_rag_chain = RunnableWithMessageHistory(
                    st.session_state["rag_chain"],
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
                response = conversational_rag_chain.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": session_id}},
                )
                answer = response["answer"]
                st.write("Answer:", answer)
                chat_history = st.session_state["store"].get(session_id, ChatMessageHistory()).messages
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Only show the history box if there is chat history
    if chat_history:
        st.markdown("""
            <div class='history-box'>
                <div class='history-title'>History</div>
        """, unsafe_allow_html=True)
        for msg in chat_history:
            prefix = "AI" if isinstance(msg, AIMessage) else "User"
            st.write(f"**{prefix}:** {msg.content}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # End main-content

elif selected_page == "How to Use":
    st.markdown("<div class='heading-box'>How to Use</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("""
### 📌 **Welcome to the RAG Chatbot with Memory!**

Harness the power of ****AI + Retrieval**** to get precise, document-specific answers - whether you're researching, studying, or building intelligent systems.

---
                                
### 🔍 What This App Does

This chatbot uses ****Retrieval-Augmented Generation (RAG)**** with conversational memory to answer your questions based on:

- 📄 **Uploaded documents:** PDF, TXT  
- 🌐 **Webpage URLs**

It keeps track of your past questions to deliver context-aware, human-like responses.

---

### 🚀 How to Use

1. **Choose a Data Source:**  
   • Upload a file using the File Uploader, **or**  
   • Paste a webpage link in the URL box

2. **Load the Content:**  
   • Click **"Load File"** or **"Load URL"**  
   • The app will chunk, embed, and store the content for quick access

3. **Ask Questions:**  
   • Type your query in the chat input below  
   • Get relevant, grounded answers-instantly!

4. **Explore More:**  
   • Use the navigation bar to learn about the app, the team, and future updates

---

## _Ready to get started? Load your content and ask away!_
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_page == "About Us":
    st.markdown("<div class='heading-box'>About Us</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("""
Welcome to our RAG Chatbot - an intelligent assistant that bridges human curiosity and machine knowledge through cutting-edge AI.

We are a passionate team of developers, designers, and researchers dedicated to making information retrieval smarter, faster, and more contextual. This project was born out of the need for a tool that can intelligently search and explain content from documents and the web - while remembering the flow of conversation, just like a human would.

---

### 🎯 Our Mission

To make AI more human-centric by combining advanced language models with intuitive user interfaces and real-world usability.

We aim to simplify how people interact with information by embedding AI in everyday research, education, and documentation workflows.

---

### 💼 What We Do

- Build LLM-powered apps with real-world utility  
- Craft sleek and simple UI experiences using Streamlit  
- Apply retrieval techniques like vector search with LangChain and Chroma  
- Continuously explore the boundaries of AI, UX, and automation

---

### 🌍 Why This Matters

In a world overflowing with unstructured information, our chatbot serves as a personal researcher, quickly digging through content and providing meaningful answers - with memory, clarity, and reliability.

Whether you're:

- A student studying from PDFs  
- A professional exploring research papers  
- Or a curious mind with big questions...

## **This app is for you.**
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_page == "Team":
    st.markdown("<div class='heading-box'>Meet the Team</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("""
                
- 👨‍💻 ****Om Kumar Singh**** – Lead Developer  
- 🧑‍💻 Ashi Jain  
- 🧑‍💻 Vansh Jain  
- 🧑‍💻 Roshmik Agrawal  
- 🧑‍💻 Aadish Chaturvedi  
- 🧑‍💻 Shambhavi Dubey
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_page == "Contact Us":
    st.markdown("<div class='heading-box'>Contact Us</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("""
📞 Phone: +91-7004918026  
✉️ Email: as120171.omkumar@gmail.com  
📷 Instagram: [@omsingh031](https://www.instagram.com/omsingh031/)  
💻 GitHub: [omsingh031](https://github.com/omsingh031)  
🔗 LinkedIn: [omsingh031](https://linkedin.com/in/omsingh031)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_page == "Future Enhancements":
    st.markdown("<div class='heading-box'>Future Enhancements</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("""
- Add support for more file types  
- Improve long-term context handling  
- Integrate authentication and user profiles  
- Export chat logs and document references
    """)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
    © 2025 Om Kumar Singh - All rights reserved.
</div>
""", unsafe_allow_html=True)
