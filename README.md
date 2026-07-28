# 💬 DocTalkAI – AI-Powered PDF Chatbot

An AI-powered PDF chatbot that enables users to upload PDF documents and interact with them using natural language. The application leverages Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware responses using Google's Gemini Flash model.

---
## 🌐 Live Demo

**Try the application here:**  
https://doctalkai-pdf-chatbot.streamlit.app/

---

## 📌 Overview

DocTalkAI is a Retrieval-Augmented Generation (RAG) application that allows users to chat with PDF documents in an intuitive conversational interface.

Instead of searching through lengthy documents manually, users can upload any PDF and ask questions in natural language. The application retrieves the most relevant sections of the document using semantic search and generates accurate answers grounded in the uploaded content.

---

## ✨ Features

- 📄 Upload any PDF document
- 💬 Ask questions in natural language
- 🤖 AI-powered answers using Gemini Flash
- 🧠 Semantic search with Hugging Face Embeddings
- 📚 Displays retrieved document sources
- ⚡ Fast vector search using ChromaDB
- 💡 Suggested questions for quick interaction
- 🖥️ Clean and responsive Streamlit interface

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | User Interface |
| LangChain | RAG Pipeline |
| ChromaDB | Vector Database |
| Hugging Face Embeddings | Text Embeddings |
| Gemini Flash | Large Language Model |
| PyPDF | PDF Processing |

---

##  RAG Pipeline

```
PDF
   │
   ▼
PyPDFLoader
   │
   ▼
Text Splitting
   │
   ▼
Generate Embeddings
   │
   ▼
Store in ChromaDB
   │
   ▼
Similarity Search
   │
   ▼
Gemini Flash
   │
   ▼
Final Response
```

---

## 📂 Project Structure

```
DocTalkAI-PDF-Chatbot/
│
├── src/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Farzanayasmin22/DocTalkAI-PDF-Chatbot.git
```

Move into the project directory

```bash
cd DocTalkAI-PDF-Chatbot
```

Install the required packages

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Google Gemini API key

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

Run the application

```bash
streamlit run app.py
```

---

##  How It Works

1. Upload a PDF document.
2. The document is split into smaller text chunks.
3. Embeddings are generated using Hugging Face Embeddings.
4. Chunks are stored in ChromaDB.
5. User questions are converted into embeddings.
6. ChromaDB retrieves the most relevant chunks.
7. Gemini Flash generates an answer using only the retrieved context.
8. Retrieved document sources are displayed for transparency.

---

## 📖 Example Questions

- Summarize this document.
- What are the main topics discussed?
- Explain the important concepts.
- What are the key points?
- Give me a brief overview of this document.

---

## 🔮 Future Improvements

- Support multiple PDF documents
- Conversation memory
- Chat history export
- PDF highlighting for retrieved answers
- OCR support for scanned PDFs
- Support for DOCX and TXT files

---

## Author

**Farzana Yasmin**

Aspiring Data Scientist | Machine Learning Enthusiast | AI & RAG Developer

---

## 📄 License

This project is licensed under the MIT License.
