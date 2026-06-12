import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize
import nltk
import ollama

nltk.download('punkt')

# -------------------------
# INIT CHROMADB
# -------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text"
)

collection = chroma_client.get_or_create_collection(
    name="my_first_collection",
    embedding_function=ollama_ef
)

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📄 PDF RAG Chatbot")

query = st.text_input("Ask a question about your PDF:")

# -------------------------
# SEARCH + LLM PIPELINE
# -------------------------
if query:

    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(docs)

    prompt = f"""
    Answer ONLY using the context below.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    st.subheader("Answer")
    st.write(response["message"]["content"])

    st.subheader("Sources")

    for doc, meta in zip(docs, metas):
        st.write(f"Page: {meta.get('page', 'Unknown')}")
        st.write(doc[:300])
        st.markdown("---")