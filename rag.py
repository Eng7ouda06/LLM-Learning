import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq

# use a tiny fast model
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_docs", embedding_function=embedding_function)

documents = [
    "Mahmoud is a second year student at Zewail City studying Communications and Information Engineering.",
    "Mahmoud built a Signal Intelligence Platform called SigmaQ using Flask, React and PostgreSQL.",
    "Mahmoud is skilled in Python, C++, SQL, and machine learning.",
    "SigmaQ features real-time FFT analysis, SNR/BER estimation, and ML-based signal quality prediction.",
    "Mahmoud is looking for a software engineering or ML internship.",
]

collection.add(
    documents=documents,
    ids=[f"doc{i}" for i in range(len(documents))]
)

print("Documents stored.\n")

def ask(question):
    results = collection.query(query_texts=[question], n_results=2)
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
    print(f"Context used: {retrieved}")
    print(f"A: {response.choices[0].message.content}\n")

ask("What is Mahmoud studying?")
ask("What are Mahmoud's technical skills?")
ask("What is SigmaQ?")
ask("What is Mahmoud's GPA?")