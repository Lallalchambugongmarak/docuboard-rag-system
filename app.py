import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="DocuBot")
st.title("📚Welcome To DocuBot📚 Know Your Files and Documents")
st.write("🤖 Can I help you? Please upload your documents and files here!")

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
        st.success(f"Indexed {len(texts)} chunks from {len(uploaded_files)} PDFs!")
    else:
        st.success(f"Indexed {st.session_state.vectorstore.index.ntotal} chunks!")

# --- VOICE + TEXT QUESTION SECTION ---
st.subheader("Ask Your Question with Voice or Text")

#----- ROW 1 - Type -----
st.write("Type a Question")
question_text = st.text_input("Type a Question", placeholder="What is document about?", label_visibility="collapsed")

st.write("") #empty space

#----- ROW 2 - Voice -----
st.write("Speak Your Question")
audio_value = st.audio_input("Speak Your Question", label_visibility="collapsed")

#----- ROW 3 - Button -----
if st.button("Ask", use_container_width=True):
    if question_text:
        st.write(f"You Asked: {question_text}")
        # put your answer code here
elif audio_value:
    st.write("Your is voice proccessing...")
    # put your voice to text code here

# If Voice is given, convert voice to text
final_query = ""
if audio_value:
    with st.spinner("Listening..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_value.getvalue())
            tmp_path = tmp.name
        try:
            r = sr.Recognizer()
            with sr.AudioFile(tmp_path) as source:
                audio = r.record(source)
                final_query = r.recognize_google(audio)
            st.info(f"You said: {final_query}")
        except Exception as e:
            st.error(f"Voice not recognized: {e}")
            os.remove(tmp_path)
final_value = final_query if audio_value else question_text
if final_value:
    if st.session_state.vectorstore is None:
        st.warning("Please upload PDFs first")
    else:
        docs = st.session_state.vectorstore.similarity_search(final_value, k=3)
        answer_text = "\n\n".join([d.page_content for d in docs])

        st.markdown("### Answer (from your documents):")
        st.markdown(answer_text)
        st.markdown("---")
        for i, d in enumerate(docs):
            st.caption(f"Source {i+1}: {d.page_content[:1000]}...")

        #--- Voice Answer ---
import re

def clean_for_voice(text):
    text = str(text)
    if "Source" in text:
        text = text.split("Source")[0]
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[*#_{}\[\]()@\"']", " ", text)
    text =re.sub(r"\s+", " ", text).strip()
    if len(text) < 10:
        text = "I found your file. Please ask mea question about your study topic, and I will explain it step by step."
    return text[:500]

    
    lines = []
    for line in text.split("\n"):
        if line.strip().startswitch(("def", "@app", "return", "http", "import", "from", "elif", "else", "if msg")):
            continue
        if "colab.research" in line or "def " in line or "@app.post" in line:
            continue
        lines.append(line)
    clean = " ".join(lines)
    clean = re.sub(r"[*#_{}\[\]()@\"']", " ", clean)
    clean =re.sub(r"\s+", " ", clean).strip()
    return text[:500]

    #-------Auto Play With Clean Text-------
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""
if final_value != st.session_state.last_spoken:
    st.session_state.last_spoken = final_value
    voice_text = clean_for_voice(final_value)
    voice_file = speak_auto(voice_text)
if voice_file:
            st.audio(voice_file, format="audio/mp3", autoplay=True)
else:
    st.warning("Voice is busy. Please wait 10 seconds")
