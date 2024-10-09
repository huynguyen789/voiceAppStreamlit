import streamlit as st
from openai import OpenAI
import io
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import pyaudio
import wave
import tempfile
import time
from audio_recorder_streamlit import audio_recorder

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

def record_audio(duration=10, sample_rate=44100):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    start_time = time.time()

    # Create a placeholder for the timer
    timer_placeholder = st.empty()

    for i in range(0, int(sample_rate / CHUNK * duration)):
        if not st.session_state.is_recording:
            break
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Update the timer every 0.1 seconds
        if i % 10 == 0:
            elapsed_time = time.time() - start_time
            timer_placeholder.text(f"Recording duration: {elapsed_time:.2f} seconds")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save as WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        wf = wave.open(tmpfile.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    st.session_state.audio_file = tmpfile.name
    st.session_state.is_recording = False

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

def get_audio_devices():
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    return input_devices

def main():
    st.title("Audio Recorder with OpenAI Whisper Speech-to-Text and GPT-4 Summary")

    st.header("1. Audio Recording")
    audio_value = st.experimental_audio_input("Record a voice message")

    if audio_value:
        # st.audio(audio_value)
        st.write("Audio recorded successfully!")
        st.session_state.audio_file = audio_value

    st.header("2. OpenAI Whisper Speech-to-Text")
    if st.button("Transcribe Audio"):
        if st.session_state.audio_file:
            audio_bytes = st.session_state.audio_file.getvalue()
            
            # Calculate and display file size in MB
            file_size_mb = len(audio_bytes) / (1024 * 1024)
            st.write(f"Audio file size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 25:
                st.write("Audio file is larger than 25 MB. Splitting into chunks...")
                chunks = split_audio(audio_bytes)
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
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "recording.wav"
                st.session_state.transcript = transcribe_audio(client, audio_file)
            
            # Display the transcript
            st.write("OpenAI Whisper result:")
            st.write(st.session_state.transcript)
        else:
            st.warning("Please record audio before transcribing.")

    # Add a button to generate summary
    if st.button("Generate Summary"):
        if st.session_state.transcript:
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarize_transcript(client, st.session_state.transcript)
        else:
            st.warning("Please record and transcribe audio before generating a summary.")

    # Display the summary if available
    if st.session_state.summary:
        st.write("GPT-4 Summary:")
        st.write(st.session_state.summary)

if __name__ == "__main__":
    main()