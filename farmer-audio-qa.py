import streamlit as st
import os
from google import genai
from google.genai import types
import time

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Farmer Audio Q&A (Gemini)", layout="wide")
st.title("🎧 Farmer Audio Q&A via Gemini")

if "history" not in st.session_state:
    st.session_state.history = []

# Step 1: Audio Upload
audio_file = st.file_uploader("Upload farmer interview (MP3/WAV)", type=["mp3", "wav"])
if audio_file:
    with open(audio_file.name, "wb") as f:
        f.write(audio_file.read())

    st.info("📤 Uploading audio file to Gemini...")
    myfile = client.files.upload(file=audio_file.name)  # Supported by SDK :contentReference[oaicite:1]{index=1}

    st.info("🔄 Transcribing via Gemini...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Please transcribe this farmer interview with speaker labels and timestamps.",
            myfile
        ]
    )
    st.session_state.transcript = response.text
    st.success("✅ Transcription complete.")
    st.text_area("Transcript:", response.text[:2000] + "...", height=300)

# Step 2: Q&A
if "transcript" in st.session_state:
    question = st.text_input("Ask a question in English:")
    if question:
        st.info("🧠 Thinking...")
        start = time.time()
        qa_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                st.session_state.transcript,
                question
            ]
        )
        answer = qa_resp.text
        t = round(time.time() - start, 2)
        st.subheader("📌 Answer")
        st.write(answer)
        st.caption(f"⏱ Responded in {t} sec")
        st.session_state.history.append((question, answer, t))

# Step 3: Q&A History
if st.session_state.history:
    st.markdown("## 🗂 Q&A History")
    for i, (q, a, t) in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A:** {a}")
        st.caption(f"Took {t} sec")
        st.markdown("---")
