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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile
import requests  # Import the requests library

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

# Define Chroma DB directory
CHROMA_DB_DIR = "chroma_db"

# Initialize Gemini models
try:
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                   convert_system_message_to_human=True)
except Exception as e:
    st.error(f"Error initializing Gemini models: {e}")
    st.stop()


# Implement session handling
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]


# Streamlit UI
st.set_page_config(page_title="RAG with Conversational Memory", layout="wide")

# Custom CSS for styling
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
        .team-member {
            font-size: 16px;
            margin-bottom: 5px;
        }
        .chat-history {
            background-color: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 8px;
            padding: 10px;
            margin-top: 20px;
        }
        .team-box {
            background-color: #ADD8E6; /* Light blue */
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main content area
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("RAG with Conversational Memory")

    # Sidebar for source selection
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

        # Display team members in a light blue box
        st.markdown("<div class='team-box'>", unsafe_allow_html=True)
        st.markdown("### Team Members: Group 31")
        st.markdown("<div class='team-member'>1. Om Kumar Singh - 23BAI10076</div>", unsafe_allow_html=True)
        st.markdown("<div class='team-member'>2. Roshmik Agrawal - 23BAI10014</div>", unsafe_allow_html=True)
        st.markdown("<div class='team-member'>3. Shambhavi Dubey - 23BAI10405</div>", unsafe_allow_html=True)
        st.markdown("<div class='team-member'>4. Vansh Jain - 23BAI10078</div>", unsafe_allow_html=True)
        st.markdown("<div class='team-member'>5. Ashi Jain - 23BAI10311</div>", unsafe_allow_html=True)
        st.markdown("<div class='team-member'>6. Aadish Chaturvedi - 23BAI11367</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Load existing vectorstore or create a new one
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

    # Rest of your code related to processing data and asking questions
    if (source_option == "Web URL" and url and load_url_button) or (
            source_option == "Upload File" and uploaded_file and load_file_button):

        try:
            if source_option == "Web URL":
                try:  # Handle potential connection errors
                    loader = WebBaseLoader(web_paths=(url,))  # REMOVED bs_kwargs
                    doc = loader.load()

                    # Extract text content
                    text_content = ""
                    for element in doc:
                        text_content += element.page_content + "\n"

                    # Create a dummy document object
                    class DummyDocument:
                        def __init__(self, page_content, metadata=None):
                            self.page_content = page_content
                            self.metadata = metadata if metadata is not None else {}  # Add metadata attribute

                    doc = [DummyDocument(text_content, metadata={"source": url})]  # Wrap in a list, add source URL

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to URL: {e}")
                    st.stop()

            elif source_option == "Upload File":
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False,
                                                  suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                file_type = uploaded_file.type
                if "text" in file_type:
                    loader = TextLoader(temp_file_path)
                elif "pdf" in file_type:
                    loader = PyPDFLoader(temp_file_path)  # Load PDF
                doc = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(doc)

            # Check if splits is empty
            if not splits:
                st.error(
                    "No content was extracted from the URL or file. Please check the URL/file and the settings.")
                st.stop()

            # Add documents to vector database
            if 'vectorstore' not in st.session_state:
                st.session_state['vectorstore'] = Chroma.from_documents(documents=splits, embedding=gemini_embeddings,
                                                     persist_directory=CHROMA_DB_DIR)
                st.write("Initialized new Chroma DB.")
            else:
                st.session_state['vectorstore'].add_documents(documents=splits)
                st.write("Added documents to existing Chroma DB.")

            retriever = st.session_state['vectorstore'].as_retriever()

            # Create RAG chain
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question "
                "If you don't know the answer, say that you don't know."
                "Use three sentences maximum and keep the answer concise."
                "\n\n"
                "{context}"
            )
            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answering_chain)

            # Implement history-aware retriever
            retriever_prompt = (
                "Given a chat history and the latest user question which might reference context in the chat history,"
                "formulate a standalone question which can be understood without the chat history."
                "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", retriever_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(model, retriever,
                                                                       contextualize_q_prompt)

            # Create conversational RAG chain
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            st.session_state["rag_chain"] = rag_chain  # Store rag_chain in session state

        except Exception as e:
            st.error(f"Error loading or processing the URL/file: {e}")

    # Asking question section
    user_question = st.text_input("Enter your question:", key="user_question")
    ask_button = st.button("Ask Question")

    if ask_button:
        if "rag_chain" in st.session_state:
            rag_chain = st.session_state["rag_chain"]
            session_id = "default_session"  # You can implement more complex session management
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
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

            # Display the answer
            st.write("Answer:", answer)

            # Display chat history
            st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
            for message in st.session_state["store"].get(session_id, ChatMessageHistory()).messages:
                if isinstance(message, AIMessage):
                    prefix = "AI"
                else:
                    prefix = "User"
                st.write(f"{prefix}: {message.content}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please load data first before asking questions.")
    st.markdown("</div>", unsafe_allow_html=True)
