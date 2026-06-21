# LLM Learning Projects

Building LLM-powered applications from scratch using Groq, ChromaDB, and Python.

## Projects

### 1. Stateful Chatbot
A conversational AI chatbot with persistent memory across the conversation.
- **Stack:** Python, Groq API, Llama 3.3 70B
- **Key concept:** Conversation history management

### 2. PDF RAG System
Ask questions about any PDF document using Retrieval Augmented Generation.
- **Stack:** Python, Groq API, ChromaDB, SentenceTransformers, PyPDF
- **Key concept:** Vector embeddings, semantic search, RAG pipeline

## Setup
```bash
pip install groq chromadb sentence-transformers pypdf python-dotenv
```
Add your Groq API key to `.env`:
```
GROQ_API_KEY=your-key-here
```