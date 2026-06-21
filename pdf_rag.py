import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from groq import Groq

# ── Setup ──
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("pdf_docs", embedding_function=embedding_function)

# ── 1. Load and chunk PDF ──
def load_pdf(path):
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # split each page into chunks of ~500 chars
            for i in range(0, len(text), 500):
                chunks.append(text[i:i+500])
    return chunks

# ── 2. Store chunks in vector DB ──
def index_pdf(path):
    chunks = load_pdf(path)
    collection.add(
        documents=chunks,
        ids=[f"chunk{i}" for i in range(len(chunks))]
    )
    print(f"Indexed {len(chunks)} chunks from PDF.\n")

# ── 3. Ask questions ──
def ask(question):
    results = collection.query(query_texts=[question], n_results=3)
    retrieved = results["documents"][0]
    context = "\n".join(retrieved)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Answer based only on this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )

    print(f"Q: {question}")
    print(f"A: {response.choices[0].message.content}\n")

# ── 4. Run ──
index_pdf("document.pdf")

print("PDF loaded! Ask me anything about it. Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    ask(question)