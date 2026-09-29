import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="DocuBot - Ask Your Company Documents")
st.title("DocuBot - Ask Your Company Documents")
st.write("Live RAG Demo | Tech: Streamlit, LangChain, FAISS, HuggingFace")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Index ONLY if new files and not already indexed
if uploaded_files:
    # Check if we already indexed these exact files
    file_names = [f.name for f in uploaded_files]
    if 'last_files' not in st.session_state or st.session_state.last_files != file_names:
        with st.spinner("Indexing documents..."):
            all_texts = []
            for pdf in uploaded_files:
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                all_texts.extend(chunks)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(all_texts, embeddings)
            st.session_state.last_files = file_names
            
        st.success(f"Indexed {len(all_texts)} chunks from {len(uploaded_files)} PDFs! Ask below.")
    else:
        st.success(f"Indexed {st.session_state.vectorstore.index.ntotal} chunks from {len(uploaded_files)} PDFs! Ask below.")

# Ask question
query = st.text_input("Ask a question", placeholder="e.g., What document is this?")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload PDFs first")
    else:
        with st.spinner("Searching..."):
            docs = st.session_state.vectorstore.similarity_search(query, k=3)
            
            # Build answer as paragraph
            answer = "\n\n".join([doc.page_content for doc in docs])
            
            st.markdown("### Answer (from your documents):")
            st.markdown(answer)
            
            st.markdown("---")
            for i, doc in enumerate(docs):
                st.write(f"**Source {i+1}:** {doc.page_content[:150]}...")
