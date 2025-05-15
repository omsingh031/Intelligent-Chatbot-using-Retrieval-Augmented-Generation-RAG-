from dotenv import load_dotenv 
import os
import streamlit as st
import warnings
import tempfile
import requests

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain import hub
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
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# State init
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "store" not in st.session_state:
    st.session_state["store"] = {}

CHROMA_DB_DIR = "chroma_db"

# Load Gemini models
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

# UI setup
st.set_page_config(page_title="RAG with Conversational Memory", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
        .sidebar .sidebar-content { background-color: #e1e8f2; padding: 20px; border-radius: 10px; }
        .stTextInput>div>div>input { border-radius: 8px; border: 1px solid #ced4da; }
        .stButton>button {
            background-color: #007bff; color: white;
            border-radius: 8px; border: none;
            padding: 10px 20px;
        }
        .stButton>button:hover { background-color: #0056b3; }
        .chat-history {
            background-color: #ffffff; border: 1px solid #ced4da; border-radius: 8px;
            padding: 10px; margin-top: 20px; max-height: 400px; overflow-y: auto;
        }
        .contact-box {
            background-color: #2f3b45; padding: 15px; border-radius: 10px;
            margin-top: 20px; margin-bottom: 10px; color: white;
        }
        .contact-title {
            font-size: 24px; font-weight: bold; color: #1a73e8; margin-bottom: 10px;
        }
        .contact-item { margin-bottom: 8px; }
        .contact-icon { margin-right: 8px; }
        a.contact-link { color: #a8cdf0; text-decoration: none; }
        a.contact-link:hover { text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

nav_options = ["Home", "About Us", "Team", "Contact Us", "Future Enhancements"]
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Home"

cols = st.columns(len(nav_options))
for i, option in enumerate(nav_options):
    if cols[i].button(option):
        st.session_state["nav_selection"] = option

selected_page = st.session_state["nav_selection"]

with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    if selected_page == "Home":
        st.title("RAG with Conversational Memory")
        st.markdown("""
        Use the sidebar to upload files or enter a URL to scrape documents. 
        Then, ask questions based on the uploaded content. The app remembers previous queries!
        """)

        with st.sidebar:
            st.markdown("<div class='sidebar sidebar-content'>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)

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
            except Exception as e:
                st.error(f"Error loading data: {e}")

        user_question = st.text_input("Enter your question:")
        if st.button("Ask"):
            if "rag_chain" not in st.session_state:
                st.warning("Please load a document or URL first.")
            else:
                try:
                    session_id = "default_session"
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
                    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
                    for msg in st.session_state["store"].get(session_id, ChatMessageHistory()).messages:
                        prefix = "AI" if isinstance(msg, AIMessage) else "User"
                        st.write(f"{prefix}: {msg.content}")
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    elif selected_page == "About Us":
        st.header("About Us")
        st.markdown("We are a team building AI chatbots powered by Retrieval-Augmented Generation with memory.")

    elif selected_page == "Team":
        st.header("Meet the Team")
        st.markdown("- Om Kumar Singh — Lead Developer\n- Other teammates...")

    elif selected_page == "Contact Us":
        st.header("Contact Us")
        st.markdown("""
        📞 Phone: +91-7004918026  
        ✉️ Email: as120171.omkumar@gmail.com  
        📷 Instagram: [@omsingh031](https://www.instagram.com/omsingh031/)  
        💻 GitHub: [omsingh031](https://github.com/omsingh031)  
        🔗 LinkedIn: [omsingh031](https://linkedin.com/in/omsingh031)
        """)

    elif selected_page == "Future Enhancements":
        st.header("Future Enhancements")
        st.markdown("""
        - Add support for more file types
        - Improve long-term context handling
        - Integrate authentication and user profiles
        - Export chat logs and document references
        """)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
  
# Footer
st.markdown("""
<div class='footer'>
    © 2025 Om Kumar Singh — All rights reserved.
</div>
""", unsafe_allow_html=True)
