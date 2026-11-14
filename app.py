
import streamlit as st
import asyncio
import edge_tts
import tempfile
import os
import re
import io
import speech_recognition as sr

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DocuBot")
st.title("📚 Welcome to DocuBot")
st.write("🤖 Upload your documents and ask questions!")


# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_files" not in st.session_state:
    st.session_state.last_files = []

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# ---------------- TEXT CLEANING ----------------
def clean_for_voice(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[*#_{}\[\]@\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000]


# ---------------- TEXT TO SPEECH ----------------
def speak_auto(text):
    text = clean_for_voice(text)

    if not text:
        return None

    async def generate_audio():
        communicate = edge_tts.Communicate(
            text,
            voice="en-US-AriaNeural"
        )

        audio_data = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])

        return bytes(audio_data)

    try:
        return asyncio.run(generate_audio())
    except Exception as e:
        st.error(f"Voice generation error: {e}")
        return None


# ---------------- SPEECH TO TEXT ----------------
def transcribe_audio(audio):
    if audio is None:
        return ""

    try:
        recognizer = sr.Recognizer()

        with sr.AudioFile(io.BytesIO(audio.getvalue())) as source:
            recorded_audio = recognizer.record(source)

        return recognizer.recognize_google(recorded_audio)

    except Exception as e:
        st.error(f"Voice recognition failed: {e}")
        return ""


# ---------------- PDF UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    file_names = [f.name for f in uploaded_files]

    if file_names != st.session_state.last_files:

        with st.spinner("Indexing PDF documents..."):

            texts = []

            for pdf in uploaded_files:
                reader = PdfReader(pdf)

                raw = "".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_text(raw)
                texts.extend(chunks)

            if texts:

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vectorstore = FAISS.from_texts(
                    texts,
                    embeddings
                )

                st.session_state.last_files = file_names

                st.success(
                    f"Indexed {len(texts)} PDF chunks!"
                )

            else:
                st.error("No readable text found in PDFs.")


# ---------------- ASK QUESTION ----------------
st.subheader("🎤 Ask DocuBot")

question_text = st.text_input(
    "Type your question",
    placeholder="What is this document about?"
)

audio_value = st.audio_input("Or speak your question")


if st.button("Ask DocuBot", use_container_width=True):

    # Choose text or voice question
    if audio_value:
        query = transcribe_audio(audio_value)
    else:
        query = question_text.strip()

    if not query:
        st.warning("Please type or speak a question.")

    elif st.session_state.vectorstore is None:
        st.error("Please upload PDFs first.")

    else:

        with st.spinner("Searching your documents..."):

            docs = st.session_state.vectorstore.similarity_search(
                query,
                k=3
            )

            if docs:

                # Retrieved document context
                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                answer = (
                    f"Your question: {query}\n\n"
                    f"According to your documents:\n\n"
                    f"{context}"
                )

                st.session_state.answer = answer

                # Generate voice answer
                with st.spinner("Generating voice answer..."):

                    audio_bytes = speak_auto(answer)

                    st.session_state.audio_bytes = audio_bytes

            else:
                st.session_state.answer = (
                    "I could not find relevant information "
                    "in your documents."
                )
                st.session_state.audio_bytes = speak_auto(
                    st.session_state.answer
                )


# ---------------- DISPLAY ANSWER ----------------
if st.session_state.answer:

    st.markdown("### 📖 DocuBot Answer")

    st.write(st.session_state.answer)

    if st.session_state.audio_bytes:

        st.markdown("### 🔊 Voice Answer")

        st.audio(
            st.session_state.audio_bytes,
            format="audio/mp3",
            autoplay=True
        )


# ---------------- THANK YOU ----------------
st.divider()

if st.button("❤️ Yes! Thank You", use_container_width=True):

    thank_you_reply = (
        "You are most welcome! "
        "Thank you for asking me. "
        "I am always happy to help you "
        "with your PDF documents."
    )

    st.success(thank_you_reply)

    thank_you_audio = speak_auto(thank_you_reply)

    if thank_you_audio:
        st.audio(
            thank_you_audio,
            format="audio/mp3",
            autoplay=True
        )

