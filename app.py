import streamlit as st
from vector_db import index_pdf, rag_answer, delete_pdf
import os

st.title("📄 RAG PDF Assistant")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

#streamlit uses uploader which is like a file in memory(a blob), but backend uses file_path so you need to convert it
def save_uploaded_file(uploaded_file):
    save_path = os.path.join("uploads", uploaded_file.name)

    os.makedirs("uploads", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path

#connecting backend to UI
if uploaded_file is not None:
    if "file_path" not in st.session_state:
        st.session_state.file_path = save_uploaded_file(uploaded_file)

    if st.button("Index PDF"):
        index_pdf(st.session_state.file_path)
        st.success("Indexed successfully!")

# ---------------- CHAT ----------------
#chat UI
st.subheader("💬 Ask questions")

query = st.text_input("Enter your question")

if query:
    answer = rag_answer(query)
    st.write(answer)

# ---------------- DELETE ----------------
st.subheader("🗑️ Delete PDF")

file_name = st.text_input("File name to delete")

if st.button("Delete PDF"):
    delete_pdf(file_name)
    st.success("Deleted from DB")