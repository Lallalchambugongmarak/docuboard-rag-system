import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="DocuBot")
st.title("DocuBot - Ask Your Company Documents")
st.caption("Live RAG Demo | Streamlit, LangChain, FAISS, HuggingFace")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'last_files' not in st.session_state:
    st.session_state.last_files = []

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    if file_names != st.session_state.last_files:
        with st.spinner("Indexing..."):
            texts = []
            for pdf in uploaded_files:
                reader = PdfReader(pdf)
                raw = "".join([p.extract_text() or "" for p in reader.pages])
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts.extend(splitter.split_text(raw))
            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(texts, emb)
            st.session_state.last_files = file_names
        st.success(f"✅ Indexed {len(texts)} chunks from {len(uploaded_files)} PDFs!")
    else:
        st.success(f"✅ Indexed {st.session_state.vectorstore.index.ntotal} chunks!")

query = st.text_input("Ask a question", placeholder="What is this document about?")

if query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload PDFs first")
    else:
        docs = st.session_state.vectorstore.similarity_search(query, k=3)
        st.markdown("### Answer (from your documents):")
        st.markdown("\n\n".join([d.page_content for d in docs]))
        st.markdown("---")
        for i, d in enumerate(docs):
            st.caption(f"Source {i+1}: {d.page_content[:200]}...")
