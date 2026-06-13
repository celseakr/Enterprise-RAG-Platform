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

