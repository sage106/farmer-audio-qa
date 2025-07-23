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
if "transcript_english" not in st.session_state:
    st.session_state.transcript_english = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "original_language" not in st.session_state:
    st.session_state.original_language = None

# Step 1: Audio Upload
audio_file = st.file_uploader("Upload farmer interview (MP3/WAV)", type=["mp3", "wav"])

# Check if a new file is uploaded or if it's the same file
if audio_file and audio_file.name != st.session_state.uploaded_file_name:
    with open(audio_file.name, "wb") as f:
        f.write(audio_file.read())

    st.info("📤 Uploading audio file to Gemini...")
    myfile = client.files.upload(file=audio_file.name)

    # First, get the original transcription
    st.info("🔄 Processing audio file...")
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[
            """Please transcribe this farmer interview exactly as spoken.
            Include speaker labels (e.g., Interviewer:, Farmer:) and timestamps where possible.
            Also identify the language being spoken.""",
            myfile
        ]
    )
    
    st.session_state.transcript = response.text
    
    # Now get the English version
    st.info("🌐 Translating to English...")
    english_response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[
            f"""Please translate the following transcript to English if it's not already in English.
            Maintain the same format with speaker labels and timestamps.
            If it's already in English, return it as is.
            
            Transcript:
            {st.session_state.transcript}"""
        ]
    )
    
    st.session_state.transcript_english = english_response.text
    st.session_state.uploaded_file_name = audio_file.name
    
    st.success("✅ Processing complete!")

# Display transcript if available
if st.session_state.transcript_english:
    st.subheader("📝 Transcript")
    
    # Create tabs for original and English versions
    tab1, tab2 = st.tabs(["English Version", "Original Transcript"])
    
    with tab1:
        col1, col2 = st.columns([4, 1])
        with col1:
            with st.expander("View English Transcript", expanded=True):
                st.text_area("", st.session_state.transcript_english, height=400, key="transcript_english_display")
        
        with col2:
            # Download button for English transcript
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_english_{timestamp}.txt"
            
            st.download_button(
                label="📥 Download (English)",
                data=st.session_state.transcript_english,
                file_name=filename,
                mime="text/plain",
                help="Download the English transcript"
            )
    
    with tab2:
        col1, col2 = st.columns([4, 1])
        with col1:
            with st.expander("View Original Transcript", expanded=False):
                st.text_area("", st.session_state.transcript, height=400, key="transcript_original_display")
        
        with col2:
            # Download button for original transcript
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_original_{timestamp}.txt"
            
            st.download_button(
                label="📥 Download (Original)",
                data=st.session_state.transcript,
                file_name=filename,
                mime="text/plain",
                help="Download the original transcript"
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
        
        # Use English transcript for Q&A
        qa_prompt = f"""Based on the following transcript of a farmer interview, please answer this question: {question}

Transcript:
{st.session_state.transcript_english}

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
        
        # Add to history (storing in session state for persistence)
        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "time": elapsed_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Step 3: Q&A History
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 🗂 Q&A History")
    
    # Create columns for history management
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col2:
        # Download Q&A history
        if st.button("📥 Download History"):
            history_text = f"Q&A History - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n\n"
            for i, item in enumerate(st.session_state.history, 1):
                history_text += f"Question {i} (Asked at {item['timestamp']}):\n{item['question']}\n\n"
                history_text += f"Answer:\n{item['answer']}\n"
                history_text += f"Response time: {item['time']} seconds\n"
                history_text += "-"*50 + "\n\n"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download",
                data=history_text,
                file_name=f"qa_history_{timestamp}.txt",
                mime="text/plain"
            )
    
    with col3:
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Display history with search functionality
    search_term = st.text_input("🔍 Search in history:", placeholder="Type to search questions or answers...")
    
    # Filter history based on search
    filtered_history = st.session_state.history
    if search_term:
        filtered_history = [
            item for item in st.session_state.history
            if search_term.lower() in item['question'].lower() or search_term.lower() in item['answer'].lower()
        ]
    
    # Display filtered history
    for i, item in enumerate(filtered_history, 1):
        with st.expander(f"Q{i}: {item['question'][:100]}..." if len(item['question']) > 100 else f"Q{i}: {item['question']}"):
            st.markdown(f"**Asked at:** {item['timestamp']}")
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.caption(f"Response time: {item['time']} seconds")

# Add a sidebar with instructions and stats
with st.sidebar:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload Audio**: Upload an MP3 or WAV file
    2. **Auto-Processing**: Audio is transcribed and translated to English
    3. **Download**: Download transcripts in English or original language
    4. **Ask Questions**: Ask questions about the content
    5. **Search History**: Search through previous Q&As
    """)
    
    st.markdown("### 📊 Session Stats")
    if st.session_state.transcript_english:
        word_count = len(st.session_state.transcript_english.split())
        st.metric("Transcript Words", f"{word_count:,}")
    
    if st.session_state.history:
        st.metric("Questions Asked", len(st.session_state.history))
        avg_time = sum(item['time'] for item in st.session_state.history) / len(st.session_state.history)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    st.markdown("### ℹ️ Features")
    st.markdown("""
    - ✅ Automatic English translation
    - ✅ Persistent Q&A history
    - ✅ Search functionality
    - ✅ Download options for all data
    - ✅ Original & translated versions
    """)
    
    # Add clear all button
    if st.button("🗑️ Clear All Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
