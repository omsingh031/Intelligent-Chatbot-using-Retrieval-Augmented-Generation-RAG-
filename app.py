from dotenv import load_dotenv 
import os
import streamlit as st
import warnings

warnings.filterwarnings('ignore')
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile
import requests

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

CHROMA_DB_DIR = "chroma_db"

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

st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #e1e8f2;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #ced4da;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .chat-history {
            background-color: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 8px;
            padding: 10px;
            margin-top: 20px;
        }
      .contact-box {
            background-color: #2f3b45;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("RAG with Conversational Memory")

    st.markdown("""
    ### 🤖 About This App

    This chatbot uses **RAG (Retrieval-Augmented Generation)** with **Google Gemini** models to provide intelligent, context-aware answers based on your uploaded file or web page content. It also remembers previous questions using **conversational memory**.

    ---

    ### 🧬 How to Use

    1. **Choose your data source** from the sidebar:
        - 📄 Upload a `.txt` or `.pdf` file
        - 🌐 Enter a valid **URL** of a webpage to scrape

    2. Click **"Load URL"** or **"Load File"** to ingest the content.

    3. Once the data is loaded, enter a question in the input box and click **"Ask Question"**.

    4. The chatbot will respond using the relevant information from your data.

    5. You can see your **chat history** below the answer.

    ---

    ⚠️ *Note: If you ask a question before loading a data source, the bot won’t have context to respond properly.*

    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div class='sidebar sidebar-content'>", unsafe_allow_html=True)
        st.header("Data Source")
        source_option = st.radio("Select a source:", ("Web URL", "Upload File"))

        if source_option == "Web URL":
            url = st.text_input("Enter the URL to scrape:")
            load_url_button = st.button("Load URL")
        elif source_option == "Upload File":
            uploaded_file = st.file_uploader("Upload a text file or PDF:", type=["txt", "pdf"])
            load_file_button = st.button("Load File")
        else:
            url = None
            uploaded_file = None
            load_url_button = False
            load_file_button = False

        st.markdown("""
        <div class='contact-box'>
            <div class='contact-title' >Contact Us</div>
            <div class='contact-item'>
                <span class='contact-icon'>📞</span>
                <a class='contact-link' href='tel:7004918026'>+91-7004918026</a>
            </div>
            <div class='contact-item'>
                <span class='contact-icon'>✉️</span>
                <a class='contact-link' href='mailto:as120171.omkumar@gmail.com'>as120171.omkumar@gmail.com</a>
            </div>
            <div class='contact-item'>
                <span class='contact-icon'>📷</span>
                <a class='contact-link' href='https://www.instagram.com/omsingh031/' target='_blank'>Instagram</a>
            </div>
            <div class='contact-item'>
                <span class='contact-icon'>💻</span>
                <a class='contact-link' href='https://github.com/omsingh031' target='_blank'>GitHub</a>
            </div>
            <div class='contact-item'>
                <span class='contact-icon'>🔗</span>
                <a class='contact-link' href='https://linkedin.com/in/omsingh031' target='_blank'>LinkedIn</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    try:
        if 'vectorstore' in st.session_state:
            vectorstore = st.session_state['vectorstore']
            st.write("Loaded existing Chroma DB from session state.")
        else:
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=gemini_embeddings)
            st.session_state['vectorstore'] = vectorstore
            st.write("Initialized new Chroma DB and stored in session state.")
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
                        self.metadata = metadata if metadata is not None else {}

                doc = [DummyDocument(text_content, metadata={"source": url})]

            elif source_option == "Upload File":
                with tempfile.NamedTemporaryFile(delete=False,
                                                  suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                file_type = uploaded_file.type
                loader = TextLoader(temp_file_path) if "text" in file_type else PyPDFLoader(temp_file_path)
                doc = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(doc)

            if not splits:
                st.error("No content was extracted. Check your input.")
                st.stop()

            if 'vectorstore' not in st.session_state:
                st.session_state['vectorstore'] = Chroma.from_documents(documents=splits, embedding=gemini_embeddings,
                                                                         persist_directory=CHROMA_DB_DIR)
                st.write("Initialized new Chroma DB.")
            else:
                st.session_state['vectorstore'].add_documents(documents=splits)
                st.write("Added documents to existing Chroma DB.")

            retriever = st.session_state['vectorstore'].as_retriever()

            system_prompt = (
                "You are an assistant for question-answering tasks. Use the retrieved context."
                "If you don't know the answer, say that you don't. Keep answers concise.\n\n{context}"
            )

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            qa_chain = create_stuff_documents_chain(model, chat_prompt)
            rag_chain = create_retrieval_chain(retriever, qa_chain)

            retriever_prompt = (
                "Given chat history and latest user question, rephrase it standalone."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", retriever_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            st.session_state["rag_chain"] = rag_chain

        except Exception as e:
            st.error(f"Error loading data: {e}")

    user_question = st.text_input("Enter your question:", key="user_question")
    ask_button = st.button("Ask Question")

    if ask_button:
        if "rag_chain" in st.session_state:
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
            for message in st.session_state["store"].get(session_id, ChatMessageHistory()).messages:
                prefix = "AI" if isinstance(message, AIMessage) else "User"
                st.write(f"{prefix}: {message.content}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please load data first before asking questions.")

    st.markdown("</div>", unsafe_allow_html=True)