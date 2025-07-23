import streamlit as st
import os
from google import genai
from google.genai import types
import tempfile
import time

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Farmer Audio Q&A (Gemini)", layout="wide")
st.title("🎧 Farmer Audio Q&A via Gemini")

if "history" not in st.session_state:
    st.session_state.history = []

# 1. Upload audio
audio_file = st.file_uploader("Upload farmer interview (MP3/WAV)", type=["mp3", "wav"])
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.info("🔄 Transcribing audio via Gemini...")
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()

    # If audio >20MB, use File API
    file_ref = client.files.upload(file=tmp_path)

    # Transcribe + summarize via Gemini
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            "Transcribe this audio interview with timestamps.",
            file_ref
        ]
    )
    transcript = response.text
    st.success("✅ Transcription complete!")
    st.text_area("Transcript:", transcript[:2000] + "...", height=300)

# 2. Q&A
if "transcript" in locals():
    question = st.text_input("Ask a question in English:")
    if question:
        st.info("🧠 Thinking...")
        start = time.time()
        res2 = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                transcript,
                question
            ]
        )
        answer = res2.text
        t = round(time.time() - start, 2)
        st.subheader("Answer:")
        st.write(answer)
        st.caption(f"⏱ Responded in {t} seconds")
        st.session_state.history.append((question, answer, t))

# 3. History
if st.session_state.history:
    st.markdown("## 🗂️ Q&A History")
    for i, (q, a, t) in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.caption(f"Took {t} sec")
        st.markdown("---")
