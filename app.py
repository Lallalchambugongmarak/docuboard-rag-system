import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="DocuBoard - RAG System")
st.title("DocuBot - Ask Your Company Documents")

st.markdown("**Live RAG Demo | Tech: Streamlit, LangChain, FAISS, HuggingFace**")

pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if pdfs:
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    with st.spinner("Indexing documents..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_texts(chunks, embeddings)
        st.session_state.db = db
        st.success(f"✅ Indexed {len(chunks)} chunks from {len(pdfs)} PDFs! Ask below.")

query = st.text_input("Ask a question")

if query and "db" in st.session_state:
    docs = st.session_state.db.similarity_search(query, k=3)
    st.write("### Answer (from your documents):")
    for i, doc in enumerate(docs):
        st.write(f"**Source {i+1}:** {doc.page_content[:500]}...")
else:
    if query:
        st.warning("Please upload PDFs first")
