import streamlit as st

from multi_pdf_loader import MultiPDFLoader
from multi_pdf_chunker import MultiPDFChunker
from semantic_embedder import SemanticEmbedder
from faiss_vector_store import FAISSVectorStore
from llm import GeminiLLM


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI University Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# ---------------- SAFE UI STYLE (FIXED VISIBILITY + INPUT FIX) ----------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f6f1e7;
            color: #2b2b2b;
        }

        /* Header */
        .main-header {
            text-align: center;
            font-size: 34px;
            font-weight: 700;
            color: #2a1f1a;
            margin-top: 10px;
        }

        .sub-header {
            text-align: center;
            color: #5a4a3f;
            font-size: 15px;
            margin-bottom: 20px;
        }

        /* Chat bubbles */
        .chat-box {
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
            color: #2b2b2b;
        }

        .user {
            background-color: #e9dccb;
        }

        .assistant {
            background-color: #ffffff;
            border: 1px solid #e6dccb;
        }

        /* Chat text fix */
        .chat-box p {
            color: #2b2b2b !important;
            margin: 0;
        }

        /* INPUT FIX */
        textarea {
            color: #ffffff !important;
            background-color: #3b2f2a !important;
        }

        textarea:focus {
            caret-color: #ffffff !important;
        }

        textarea::placeholder {
            color: #d6c7b2 !important;
        }

        input {
            color: #ffffff !important;
            background-color: #3b2f2a !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- HEADER ----------------
st.markdown(
    "<div class='main-header'>📘 AI University Policy Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-header'>Ask questions about academic regulations, attendance, exams, and student policies</div>",
    unsafe_allow_html=True
)


# ---------------- PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    loader = MultiPDFLoader(pdf_dir="data/pdfs")
    documents = loader.load_all()

    chunker = MultiPDFChunker(chunk_size=500, overlap=100)
    chunks, metadata = chunker.chunk_documents(documents)

    embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")
    vectors = embedder.embed(chunks)

    store = FAISSVectorStore()
    store.add_chunks(chunks, vectors)

    llm = GeminiLLM()

    return embedder, store, llm


embedder, store, llm = load_pipeline()


# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    css_class = "user" if msg["role"] == "user" else "assistant"

    st.markdown(
        f"<div class='chat-box {css_class}'>{msg['content']}</div>",
        unsafe_allow_html=True
    )


# ---------------- USER INPUT ----------------
user_query = st.chat_input("Ask a question about your documents...")

if user_query:

    # store user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # embed query
    query_vector = embedder.embed_query(user_query)

    # retrieve chunks
    results = store.search(query_vector, k=3)

    # build history
    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[-6:]
    )

    # generate answer
    answer = llm.generate_answer(
        question=user_query,
        context_chunks=results,
        history=history_text
    )

    # store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # refresh UI
    st.rerun()