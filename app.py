import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from pypdf import PdfReader
# NEW - no chains needed
from langchain_core.prompts import PromptTemplate

st.title("DocuBot - Ask Your Company Documents")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        text = ""
        for pdf in uploaded_files:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Save for later questions
        st.session_state.vstore = vectorstore
        st.success("Docs ready! Ask anything.")

question = st.text_input("Ask a question")
if question and "vstore" in st.session_state:
    retriever = st.session_state.vstore.as_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    
    llm = OpenAI(temperature=0)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    answer = llm.invoke(prompt)
    st.write("**Answer:**", answer)
