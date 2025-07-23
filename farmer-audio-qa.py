import streamlit as st
import os
from google import genai
import time

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Farmer Transcript Q&A with Gemini", layout="centered")
st.title("📖 Farmer Interview Q&A (Gemini 2.5 Pro)")

if "history" not in st.session_state:
    st.session_state.history = []

# Step 1: Upload transcript file
transcript_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])
if transcript_file:
    transcript = transcript_file.read().decode("utf-8")
    st.text_area("Transcript Preview:", transcript[:1000] + "...", height=200)

    # Step 2: Ask questions
    question = st.text_input("Ask a question in English:")
    if question:
        with st.spinner("🤖 Thinking..."):
            start = time.time()
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[transcript, question]
            )
            answer = response.text
            duration = round(time.time() - start, 2)

        st.success("📌 Answer:")
        st.write(answer)
        st.caption(f"⏱️ Response time: {duration} seconds")

        st.session_state.history.append((question, answer, duration))

# Step 3: Show history
if st.session_state.history:
    st.markdown("## 🗂️ Q&A History")
    for i, (q, a, t) in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A:** {a}")
        st.caption(f"Answered in {t} sec")
        st.markdown("---")
