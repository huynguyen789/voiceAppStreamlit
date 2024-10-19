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
MAX_CHUNK_DURATION = 300  # 5 minutes in seconds

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

def summarize_audio_chunk(chunk, mime_type):
    model = genai.GenerativeModel("gemini-1.5-flash")
    buffer = io.BytesIO()
    chunk.export(buffer, format="mp3")
    audio_bytes = buffer.getvalue()
    encoded_chunk = base64.b64encode(audio_bytes).decode('utf-8')
    content = [
        "Summarize the audio content.",
        {
            "mime_type": mime_type,
            "data": encoded_chunk
        }
    ]
    token_count = model.count_tokens(content).total_tokens
    response = model.generate_content(content)
    return response.text, token_count

def summarize_audio(audio_file):
    chunks = split_audio(audio_file, MAX_CHUNK_DURATION)
    
    summaries = []
    total_tokens = 0
    
    for chunk in chunks:
        summary, tokens = summarize_audio_chunk(chunk, audio_file.type)
        summaries.append(summary)
        total_tokens += tokens
    
    full_summary = "\n\n".join(summaries)
    return full_summary, total_tokens

st.title("Audio Summarizer with Gemini API")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "aiff", "aac", "ogg", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Summarize Audio"):
        start_time = time.time()
        with st.spinner("Summarizing audio..."):
            summary, token_count = summarize_audio(uploaded_file)
        end_time = time.time()
        processing_time = end_time - start_time
        
        st.subheader("Summary:")
        st.write(summary)
        st.subheader("Token Count:")
        st.write(f"This audio file used approximately {token_count} tokens.")
        st.subheader("Processing Time:")
        st.write(f"The summarization took {processing_time:.2f} seconds.")
