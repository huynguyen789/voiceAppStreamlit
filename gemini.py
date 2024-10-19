import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
import base64
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MAX_CHUNK_SIZE = 19 * 1024 * 1024  # 19 MB chunks
MAX_CHUNK_DURATION = 180  # 5 minutes in seconds

def split_audio(audio_file, chunk_duration):
    audio = AudioSegment.from_file(audio_file)
    total_length = len(audio)
    num_chunks = total_length // (chunk_duration * 1000) + (total_length % (chunk_duration * 1000) > 0)
    
    chunks = []
    for i in range(num_chunks):
        start_time = i * chunk_duration * 1000
        end_time = start_time + chunk_duration * 1000
        chunk = audio[start_time:end_time]
        chunks.append(chunk)
    
    return chunks

def transcribe_audio_chunk(chunk, mime_type):
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    buffer = io.BytesIO()
    chunk.export(buffer, format="mp3")
    audio_bytes = buffer.getvalue()
    encoded_chunk = base64.b64encode(audio_bytes).decode('utf-8')
    content = [
        "Transcribe the audio content accurately, get the important information.",
        {
            "mime_type": mime_type,
            "data": encoded_chunk
        }
    ]
    response = model.generate_content(content)
    return response.text

def transcribe_and_summarize_audio(audio_file):
    chunks = split_audio(audio_file, MAX_CHUNK_DURATION)
    
    transcripts = []
    
    for chunk in chunks:
        transcript = transcribe_audio_chunk(chunk, audio_file.type)
        transcripts.append(transcript)
    
    full_transcript = "\n\n".join(transcripts)
    
    # Summarize the full transcript using Gemini Flash
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    summary_prompt = f"Summarize the below audio transcript. First, have a 1-2 sentence summary of the content. Next, provide a detailed summary of the content:\n\n{full_transcript}"
    summary_response = model.generate_content(summary_prompt)
    
    return full_transcript, summary_response.text

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None

st.title("Audio Summarizer with Gemini API")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "aiff", "aac", "ogg", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Transcribe and Summarize Audio"):
        start_time = time.time()
        with st.spinner("Transcribing and summarizing audio..."):
            st.session_state.transcript, st.session_state.summary = transcribe_and_summarize_audio(uploaded_file)
        end_time = time.time()
        st.session_state.processing_time = end_time - start_time

if st.session_state.transcript is not None:
    st.subheader("Full Transcript:")
    with st.expander("Click to expand/collapse"):
        st.write(st.session_state.transcript)
    
    st.subheader("Summary:")
    st.write(st.session_state.summary)
    
    st.subheader("Processing Time:")
    st.write(f"The transcription and summarization took {st.session_state.processing_time:.2f} seconds.")
