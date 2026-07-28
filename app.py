import streamlit as st
import tempfile
import os
import uuid

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.llm import get_llm
from src.rag_pipeline import generate_answer

# ------------------ Page Configuration ------------------ #
st.set_page_config(page_title="DocTalkAI", page_icon="💬", layout="centered")

# ------------------ Header ------------------ #
st.title("💬 DocTalkAI")
st.markdown("""
### Your document. Your conversation.
**Read less. Understand more.**
Upload a PDF, ask questions, and get AI-powered answers instantly.
""")
st.caption("Powered by Gemini • LangChain • ChromaDB")
st.divider()

# ---------- Core functions ----------

embedding_model = get_embedding_model()

llm = get_llm()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None

# ------------------ Sidebar (always visible) ------------------ #
with st.sidebar:
    st.title("💬 DocTalkAI")
    st.markdown("---")
    st.subheader("⚙️ AI Stack")
    st.markdown("""
- Gemini Flash
- BAAI/bge-small-en-v1.5
- ChromaDB
- LangChain
- Streamlit
""")
    st.markdown("---")
    st.subheader("About")
    st.caption("Upload your PDF and chat with it using Retrieval-Augmented Generation (RAG).")
    st.markdown("---")

# ------------------ PDF Upload ------------------ #
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"], help="Upload a PDF document to start chatting with it.")

if uploaded_file is not None:
    file_size_kb = uploaded_file.size / 1024

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Reading and processing your document..."):
            docs = load_pdf(tmp_path)
            chunks = split_documents(docs)
            persist_dir = f"chroma_db/{st.session_state.session_id}"
            vector_store = create_vector_store(chunks, embedding_model, persist_directory=persist_dir)

        st.session_state.vector_store = vector_store
        st.session_state.processed_file = uploaded_file.name
        st.session_state.num_pages = len(docs)
        st.session_state.chat_history = []

    st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")

    # ---------- Sidebar: document details (only shown once a file exists) ----------
    with st.sidebar:
    
        st.markdown("### 📄 Current Document")
    
        with st.container(border=True):
            st.markdown(f"**📄 File:** `{uploaded_file.name}`")
            st.markdown(f"**📑 Pages:** `{st.session_state.num_pages}`")
            st.markdown(f"**💾 Size:** `{file_size_kb:.1f} KB`")
    
        st.write("")
    
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # ------------------ Chat Interface (main page) ------------------ #
    st.subheader("🤖 Chat with your document")

    # Suggested Questions
    st.markdown("### 💡 Suggested Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📑 Summarize", use_container_width=True):
            st.session_state.suggested_question = "Summarize this document."
    
    with col2:
        if st.button("📖 Main Topics", use_container_width=True):
            st.session_state.suggested_question = "What are the main topics discussed in this document?"
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🔑 Key Points", use_container_width=True):
            st.session_state.suggested_question = "What are the key points in this document?"
    
    with col4:
        if st.button("🧠 Explain Concepts", use_container_width=True):
            st.session_state.suggested_question = "Explain the important concepts in this document."

    # Display previous messages (always, regardless of suggested-question state)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks used)"):
                    for doc in msg["sources"]:
                        page = doc.metadata.get("page", "unknown")
                        page_display = page + 1 if isinstance(page, int) else page
                        st.markdown(f"**Page {page_display}**")
                        st.caption(doc.page_content[:300] + "...")

    # Chat input (also accepts a suggested-question click)
    user_question = st.chat_input("Type your question here...")

    if st.session_state.suggested_question:
        user_question = st.session_state.suggested_question
        st.session_state.suggested_question = None

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching your document..."):
                answer, sources = generate_answer(user_question, st.session_state.vector_store, llm)
                st.write(answer)
                if sources:
                    with st.expander(f"📚 Retrieved Sources (Top {len(sources)} Matches)"):
                        for doc in sources:
                            page = doc.metadata.get("page", "unknown")
                            page_display = page + 1 if isinstance(page, int) else page
                            st.markdown(f"**Page {page_display}**")
                            st.caption(doc.page_content[:300] + "...")

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

else:
    st.info("👆 Upload a PDF above to get started.")
