import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
import io
import os
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def split_audio(audio_bytes, chunk_size_mb=25):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    chunk_length_ms = int((chunk_size_mb * 1024 * 1024 * 8) / (audio.frame_rate * audio.sample_width * audio.channels))
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def transcribe_audio(client, audio_file):
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
        return transcript.text
    except Exception as e:
        return f"Error transcribing: {str(e)}"
    
def get_openai_response(client, prompt):
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def summarize_transcript(client, transcript):
    prompt = f"Please summarize the following transcript:\n\n{transcript}"
    summary = get_openai_response(client, prompt)
    return summary

def main():
    st.title("Audio Recorder with OpenAI Whisper Speech-to-Text and GPT-4 Summary")

    # Initialize session state for transcript and summary
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    st.header("1. Simple Audio Recording")
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False
    )
    if audio:
        st.audio(audio['bytes'])
        st.write("Audio recorded successfully!")

    st.header("2. OpenAI Whisper Speech-to-Text")
    whisper_audio = mic_recorder(
        start_prompt="Start Whisper recording",
        stop_prompt="Stop Whisper recording",
        just_once=True,
        key='whisper_recorder'
    )
    if whisper_audio:
        st.audio(whisper_audio['bytes'])
        st.write("Audio recorded for Whisper, transcribing...")
        
        # Calculate and display file size in MB
        file_size_mb = len(whisper_audio['bytes']) / (1024 * 1024)
        st.write(f"Audio file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 25:
            st.write("Audio file is larger than 25 MB. Splitting into chunks...")
            chunks = split_audio(whisper_audio['bytes'])
            full_transcript = ""
            for i, chunk in enumerate(chunks):
                st.write(f"Transcribing chunk {i+1}/{len(chunks)}...")
                chunk_file = io.BytesIO()
                chunk.export(chunk_file, format="mp3")
                chunk_file.seek(0)
                chunk_transcript = transcribe_audio(client, chunk_file)
                full_transcript += chunk_transcript + " "
            st.session_state.transcript = full_transcript.strip()
        else:
            audio_file = io.BytesIO(whisper_audio['bytes'])
            audio_file.name = "recording.webm"
            st.session_state.transcript = transcribe_audio(client, audio_file)
        
        # Display the transcript
        st.write("OpenAI Whisper result:")
        st.write(st.session_state.transcript)

    # Add a button to generate summary
    if st.button("Generate Summary"):
        if st.session_state.transcript:
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarize_transcript(client, st.session_state.transcript)
        else:
            st.warning("Please record and transcribe audio before generating a summary.")

    # Display the summary if available
    if st.session_state.summary:
        st.write("GPT-4o Summary:")
        st.write(st.session_state.summary)

if __name__ == "__main__":
    main()