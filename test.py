import streamlit as st
from pydub import AudioSegment
from openai import OpenAI
import os
import tempfile
import whisper

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load Whisper model
model = whisper.load_model("small.en")

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

def transcribe_audio(audio_file):
    try:
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Error transcribing: {str(e)}"

def get_openai_response(client, prompt):
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# New function to process audio
def process_audio(audio_file, file_size):
    if file_size <= 25 * 1024 * 1024:  # 25MB
        transcript = transcribe_audio(audio_file)
    else:
        chunks = split_audio(audio_file, 60)  # Split into 1-minute chunks
        transcripts = []
        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                chunk.export(temp_file.name, format="mp3")
                transcript = transcribe_audio(temp_file.name)
                transcripts.append(transcript)
            os.unlink(temp_file.name)
        transcript = " ".join(transcripts)
    
    summary_prompt = f"Please summarize the following transcript:\n\n{transcript}"
    summary = get_openai_response(client, summary_prompt)
    return transcript, summary

# Streamlit app
st.title("Audio Transcription and Summarization")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    file_size = uploaded_file.size
    
    with st.spinner("Processing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        transcript, summary = process_audio(temp_file_path, file_size)
        os.unlink(temp_file_path)
    
    st.subheader("Transcript")
    st.text_area("Full Transcript", transcript, height=200)
    
    st.subheader("Summary")
    st.write(summary)
