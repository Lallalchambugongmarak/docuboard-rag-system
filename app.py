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

# Ask a question
query = st.text_input("Ask a question", placeholder="e.g., Any important words there in the PDF?")

if query:
    if 'vectorstore' not in st.session_state:
        st.warning("⚠️ Please upload PDFs first")
    else:
        with st.spinner("Searching..."):
            # Your retrieval code here
            docs = st.session_state.vectorstore.similarity_search(query, k=3)

            # Build answer - as a SINGLE string, not loop
            answer = "\n".join([doc.page_content for doc in docs[:2]])
            source_1 = docs[0].metadata.get('source', 'Doc 1') if len(docs) > 0 else "No source"
            source_2 = docs[1].metadata.get('source', 'Doc 2') if len(docs) > 1 else "No source"

            # DISPLAY - FIXED
            st.markdown("### Answer (from your documents):")
            st.markdown(answer) # Paragraph, not vertical list

            st.markdown("---")
            st.write(f"**Source 1:** {source_1}")
            st.write(f"**Source 2:** {source_2}")
else:
    st.info("👆 Upload PDFs and ask a question above")
