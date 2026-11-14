
import streamlit as st
import asyncio
import edge_tts
import io
import re
import speech_recognition as sr

from pypdf import PdfReader
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="DocuBot AI", page_icon="📚")

st.title("📚 DocuBot AI")
st.subheader("Your Intelligent PDF Voice Assistant")
st.write("Upload PDFs, ask questions, and listen to AI answers.")

# Gemini client
try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    client = None


# ==========================================
# SESSION STATE
# ==========================================
defaults = {
    "vectorstore": None,
    "last_files": [],
    "answer": "",
    "question": "",
    "audio_bytes": None,
    "sources": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# SPEECH UTILITIES
# ==========================================
def clean_for_voice(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[*#_{}\[\]@\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:1500]


def speak_auto(text):
    text = clean_for_voice(text)

    if not text:
        return None

    async def generate():
        communicate = edge_tts.Communicate(
            text,
            voice="en-US-AriaNeural"
        )

        audio = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])

        return bytes(audio)

    try:
        return asyncio.run(generate())
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None


def transcribe_audio(audio):
    if audio is None:
        return ""

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(io.BytesIO(audio.getvalue())) as source:
            recorded = recognizer.record(source)

        return recognizer.recognize_google(recorded)

    except sr.UnknownValueError:
        st.warning("I could not understand your voice.")
        return ""

    except sr.RequestError:
        st.error("Speech recognition service unavailable.")
        return ""

    except Exception as e:
        st.error(f"Audio error: {e}")
        return ""


# ==========================================
# GEMINI AI ANSWER
# ==========================================
def generate_ai_answer(question, context):

    if client is None:
        return "Gemini API key is missing. Configure Streamlit secrets."

    prompt = f"""
You are DocuBot, an intelligent PDF document assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Explain clearly and naturally.
2. Do not invent information.
3. If the answer is missing, say it is not found in the uploaded documents.
4. Summarize long passages instead of copying everything.
5. Answer in a friendly conversational style.
6. Make the answer suitable for text-to-speech.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

DOCUBOT ANSWER:
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text or "I could not generate an answer."

    except Exception as e:
        st.error(f"Gemini error: {e}")
        return "Sorry, I could not generate an AI answer."


# ==========================================
# PDF UPLOAD + INDEXING
# ==========================================
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    file_names = [f.name for f in uploaded_files]

    if file_names != st.session_state.last_files:

        with st.spinner("Reading and indexing PDFs..."):

            all_chunks = []
            all_metadata = []

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            for pdf in uploaded_files:

                reader = PdfReader(pdf)

                for page_number, page in enumerate(reader.pages):

                    text = page.extract_text() or ""

                    chunks = splitter.split_text(text)

                    for chunk in chunks:
                        all_chunks.append(chunk)

                        all_metadata.append({
                            "source": pdf.name,
                            "page": page_number + 1
                        })

            if all_chunks:

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vectorstore = FAISS.from_texts(
                    all_chunks,
                    embeddings,
                    metadatas=all_metadata
                )

                st.session_state.last_files = file_names

                st.success(
                    f"Indexed {len(all_chunks)} chunks "
                    f"from {len(uploaded_files)} PDFs."
                )

            else:
                st.error("No readable PDF text found.")


# ==========================================
# QUESTION INPUT
# ==========================================
st.divider()
st.header("🎤 Ask DocuBot")

question_text = st.text_input(
    "Type your question",
    placeholder="Explain this document..."
)

audio_value = st.audio_input("Or speak your question")


# ==========================================
# ASK BUTTON
# ==========================================
if st.button("🚀 Ask DocuBot", use_container_width=True):

    query = (
        transcribe_audio(audio_value)
        if audio_value
        else question_text.strip()
    )

    if not query:
        st.warning("Please enter or speak a question.")

    elif st.session_state.vectorstore is None:
        st.error("Please upload PDFs first.")

    else:

        st.session_state.question = query

        with st.spinner("Searching your PDFs..."):

            docs = st.session_state.vectorstore.similarity_search(
                query,
                k=5
            )

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            st.session_state.sources = docs

        # Generate real AI answer
        with st.spinner("Gemini is thinking..."):

            answer = generate_ai_answer(
                query,
                context
            )

            st.session_state.answer = answer

        # Generate voice answer
        with st.spinner("Generating voice answer..."):

            st.session_state.audio_bytes = speak_auto(
                answer
            )


# ==========================================
# DISPLAY AI ANSWER
# ==========================================
if st.session_state.answer:

    st.divider()

    st.subheader("🤖 DocuBot AI Answer")

    st.write("Question:", st.session_state.question)

    st.markdown(st.session_state.answer)

    # AUDIO OUTPUT
    if st.session_state.audio_bytes:

        st.subheader("🔊 Listen to DocuBot")

        st.audio(
            st.session_state.audio_bytes,
            format="audio/mp3",
            autoplay=True
        )

    # SOURCES
    with st.expander("📄 View PDF Sources"):

        for i, doc in enumerate(st.session_state.sources):

            source = doc.metadata.get(
                "source",
                "Unknown PDF"
            )

            page = doc.metadata.get(
                "page",
                "Unknown page"
            )

            st.caption(
                f"Source {i+1}: {source} | Page {page}"
            )

            st.write(doc.page_content)


# ==========================================
# THANK YOU
# ==========================================
st.divider()

if st.button("❤️ Yes! Thank You", use_container_width=True):

    reply = (
        "You are most welcome! "
        "I am DocuBot, and I am always happy "
        "to help you understand your documents."
    )

    st.success(reply)

    thank_you_audio = speak_auto(reply)

    if thank_you_audio:

        st.audio(
            thank_you_audio,
            format="audio/mp3",
            autoplay=True
        )

