import streamlit as st
import os
from google import genai
from google.genai import types
import time
from datetime import datetime

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Farmer Audio Q&A (Gemini)", layout="wide")
st.title("🎧 Farmer Audio Q&A via Gemini")

# Initialize session states
if "history" not in st.session_state:
    st.session_state.history = []
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# Step 1: Audio Upload
audio_file = st.file_uploader("Upload farmer interview (MP3/WAV)", type=["mp3", "wav"])

# Check if a new file is uploaded or if it's the same file
if audio_file and audio_file.name != st.session_state.uploaded_file_name:
    with open(audio_file.name, "wb") as f:
        f.write(audio_file.read())

    st.info("📤 Uploading audio file to Gemini...")
    myfile = client.files.upload(file=audio_file.name)

    st.info("🔄 Transcribing in English via Gemini...")
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[
            """Please transcribe this farmer interview directly in English. 
            If the audio is in another language, provide the English translation as you transcribe.
            Include speaker labels (e.g., Interviewer:, Farmer:) and timestamps where possible.
            Format the transcript clearly with proper paragraphs and punctuation.""",
            myfile
        ]
    )
    
    # Store transcript and file name in session state
    st.session_state.transcript = response.text
    st.session_state.uploaded_file_name = audio_file.name
    
    st.success("✅ English transcription complete.")

# Display transcript if available
if st.session_state.transcript:
    st.subheader("📝 English Transcript")
    
    # Create columns for transcript display and download button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Display transcript in expandable text area
        with st.expander("View Full Transcript", expanded=True):
            st.text_area("", st.session_state.transcript, height=400, key="transcript_display")
    
    with col2:
        # Download button for transcript
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_english_{timestamp}.txt"
        
        st.download_button(
            label="📥 Download Transcript",
            data=st.session_state.transcript,
            file_name=filename,
            mime="text/plain",
            help="Download the full English transcript as a text file"
        )

    # Step 2: Q&A
    st.markdown("---")
    st.subheader("💬 Ask Questions")
    
    # Create a form for question input to prevent re-runs
    with st.form(key="question_form"):
        question = st.text_input("Ask a question about the interview (in English):")
        submit_button = st.form_submit_button("Ask")
    
    if submit_button and question:
        st.info("🧠 Thinking...")
        start = time.time()
        
        # Create a focused prompt for Q&A
        qa_prompt = f"""Based on the following transcript of a farmer interview, please answer this question: {question}

Transcript:
{st.session_state.transcript}

Please provide a clear and concise answer based only on the information in the transcript. If the information is not available in the transcript, please state that."""
        
        qa_resp = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=qa_prompt
        )
        
        answer = qa_resp.text
        elapsed_time = round(time.time() - start, 2)
        
        # Display answer
        st.markdown("### 📌 Answer")
        st.write(answer)
        st.caption(f"⏱ Responded in {elapsed_time} seconds")
        
        # Add to history
        st.session_state.history.append((question, answer, elapsed_time))

# Step 3: Q&A History
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 🗂 Q&A History")
    
    # Add option to download Q&A history
    if st.button("📥 Download Q&A History"):
        history_text = "Q&A History\n" + "="*50 + "\n\n"
        for i, (q, a, t) in enumerate(st.session_state.history, 1):
            history_text += f"Question {i}: {q}\n"
            history_text += f"Answer: {a}\n"
            history_text += f"Response time: {t} seconds\n"
            history_text += "-"*50 + "\n\n"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download",
            data=history_text,
            file_name=f"qa_history_{timestamp}.txt",
            mime="text/plain"
        )
    
    # Display history
    for i, (q, a, t) in enumerate(st.session_state.history, 1):
        with st.expander(f"Q{i}: {q[:100]}..." if len(q) > 100 else f"Q{i}: {q}"):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")
            st.caption(f"Response time: {t} seconds")

# Add a sidebar with instructions
with st.sidebar:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload Audio**: Upload an MP3 or WAV file of a farmer interview
    2. **Auto-Transcription**: The audio will be automatically transcribed in English
    3. **Download**: Download the English transcript using the download button
    4. **Ask Questions**: Ask questions about the interview content
    5. **View History**: See all your previous questions and answers
    """)
    
    st.markdown("### ℹ️ Features")
    st.markdown("""
    - ✅ Direct English transcription
    - ✅ Automatic translation if audio is in another language
    - ✅ One-time transcription (no re-processing)
    - ✅ Download transcript as text file
    - ✅ Download Q&A history
    - ✅ Speaker labels and timestamps
    """)
    
    # Add clear button
    if st.button("🗑️ Clear All Data"):
        st.session_state.history = []
        st.session_state.transcript = None
        st.session_state.uploaded_file_name = None
        st.rerun()
