import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
from gtts import gTTs
import tempfile
import os

st.set_page_config(page_title="DocuBot")
st.title("📚🐰Welcome To DocuBot! Know Your Paper and Documents😘")
st.caption("Please Upload your PDF Files")

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

# --- VOICE + TEXT QUESTION SECTION ---
st.divider()
st.subheader("Ask with Voice or TEXT")

col1, col2 = st.columns(2)
with col1:
    query_text = st.text_input("Ask a question", placeholder="What is document about?")
    with col2:
        audio_query = st.audio_input("Or Speak Your Question")
        final_query = query_text

# If Voice is given, convert voice to text
if audio_query:
    with st.spinner("Listening..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_query.getvalue())
            tmp_path = tmp.name
        r = sr.Recognizer()
        try:
            with sr.AudioFile(temp_path) as source:
                audio = r.record(source)
                final_query = r.recognize_google(audio)
                st.info(f"You said: {final_query}")
        except Exception as e:
            st.error(f"Voice not recognized: {e}")
            os.remove(tmp_path)
if final_query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload PDFs first")
    else:
        docs = st.session_state.vectorstore.similarity_search(final_query, k=3)
        answer_text = "\n\n".join([d.page_content for d in docs])

        st.markdown("### Answer (from your documents):")
        st.markdown(answer_text)
        st.markdown("---")
        for i, d in enumerate(docs):
            st.caption(f"Source {i=1}: {d.page_content[:1000]}...")

        #--- VOICE ANSWER ---
        try:
            tts = gTTs(text=answer_text[:4000], lang='en') # first 4000 chars only
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
                st.caption("Playing Voice Answer")
        except Exception as e:
            st.error(f"TTS Error: {e}")


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
            st.caption(f"Source {i+1}: {d.page_content[:1000]}...")
