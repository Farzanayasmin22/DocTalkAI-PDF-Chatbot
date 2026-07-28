import streamlit as st
import tempfile
import os
import uuid
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

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

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ---------- CHANGED: embedding model and LLM are now cached resources ----------
# Without @st.cache_resource, Streamlit was re-loading the full sentence-transformers
# model (and re-initializing the LLM client) on *every single rerun* of the script
# (every button click, every chat message). On Streamlit Community Cloud's shared,
# memory-limited process, that repeated loading under concurrent users is what was
# starving memory and causing the embedding step to fail with
# "Expected Embeddings to be non-empty list".

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_llm():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in the .env file.")
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0)

embedding_model = load_embedding_model()
llm = load_llm()
# ---------- END CHANGED ----------

def create_vector_store(chunks, embedding_model, persist_directory="chroma_db_app"):
    return Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_directory)

def generate_answer(query, vector_store, llm, k=3):
    retrieved_docs = vector_store.similarity_search(query, k=k)
    if not retrieved_docs:
        return (
            "I couldn't find information related to that question in the uploaded PDF.\n\n"
            "Try:\n• Rephrasing your question\n• Asking about another topic\n• Requesting a summary of a section"
        ), []

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer the user's question.
Do not use any outside knowledge.

If the user asks for a summary:
- Summarize the document in clear, well-structured paragraphs.
- Include:
  • The main purpose of the document
  • The important concepts discussed
  • The key findings or contributions
  • The overall conclusion (if available)
- Do NOT list citations, references, or bibliography unless the user specifically asks about them.
- Keep the summary concise, informative, and easy to understand.

For all other questions:
- Answer clearly using complete sentences.
- Explain the answer in a simple and detailed manner based only on the provided context.

If the answer is not found in the context, reply exactly:
"I couldn't find that information in the uploaded document."

Context = {context}
Question = {query}

Answer:
"""
    response = llm.invoke(prompt)
    answer = response.content[0]['text'] if isinstance(response.content, list) else response.content
    return answer, retrieved_docs

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
- 🤖 Gemini Flash
- 🧠 BAAI/bge-small-en-v1.5
- 📚 ChromaDB
- 🔗 LangChain
- 🎨 Streamlit
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

            # ---------- CHANGED: guard against PDFs with no extractable text ----------
            # Scanned/image-only PDFs, password-protected PDFs, or corrupted PDFs
            # produce zero usable text chunks, which previously crashed Chroma with
            # "Expected Embeddings to be non-empty list". Now we catch it early and
            # show a clear message instead.
            if not chunks or all(not c.page_content.strip() for c in chunks):
                st.error(
                    "⚠️ I couldn't extract any text from this PDF. It might be a "
                    "scanned/image-only PDF, password-protected, or corrupted. "
                    "Try a different PDF, or run OCR on it first."
                )
                st.stop()
            # ---------- END CHANGED ----------

            persist_dir = f"chroma_db/{st.session_state.session_id}"

            try:
                vector_store = create_vector_store(chunks, embedding_model, persist_directory=persist_dir)
            except Exception as e:
                # ---------- CHANGED: friendly error instead of a raw stack trace ----------
                st.error(
                    "⚠️ Something went wrong while processing this PDF. "
                    "Please try again, or try a different file."
                )
                st.exception(e)
                st.stop()
                # ---------- END CHANGED ----------

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
