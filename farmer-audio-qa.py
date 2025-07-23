import streamlit as st
import openai
import os
import tempfile
import time

# Set OpenAI API key from secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Farmer Audio Q&A", layout="centered")

st.title("🎙️ Farmer Audio Q&A System")
st.write("Upload an interview with a farmer, then ask questions about their responses.")

# Initialize session state
if "qa_count" not in st.session_state:
    st.session_state.qa_count = 0
    st.session_state.transcript = ""
    st.session_state.history = []

# 1. Upload audio file
audio_file = st.file_uploader("📤 Upload Audio File (.mp3, .wav)", type=["mp3", "wav"])

if audio_file:
    with st.spinner("🔍 Transcribing audio..."):
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Transcribe using OpenAI Whisper
        with open(tmp_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)["text"]

        st.session_state.transcript = transcript
        st.success("✅ Transcription complete.")
        st.markdown("**📝 Transcript Preview:**")
        st.write(transcript[:1000] + "...")  # show first 1000 chars

# 2. Ask questions
if st.session_state.transcript:
    question = st.text_input("💬 Ask a question about the audio:")
    if question:
        start = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant answering questions about an interview with an Indian farmer. The transcription of the interview is below."},
                {"role": "user", "content": f"Transcript:\n{st.session_state.transcript}"},
                {"role": "user", "content": question}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        duration = round(time.time() - start, 2)

        st.session_state.qa_count += 1
        st.session_state.history.append((question, answer, duration))

        st.markdown("### 🧠 Answer:")
        st.write(answer)
        st.caption(f"⏱️ Responded in {duration} seconds")

# 3. Show analytics
if st.session_state.history:
    st.markdown("## 📊 Q&A History")
    for i, (q, a, t) in enumerate(st.session_state.history, 1):
        st.markdown(f"**{i}. Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.caption(f"🕒 Took {t} sec")
        st.markdown("---")

    st.markdown(f"✅ Total Questions Answered: **{st.session_state.qa_count}**")
