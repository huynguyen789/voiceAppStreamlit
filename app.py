import streamlit as st
from openai import OpenAI
import io
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import datetime

# Load environment variables
load_dotenv()

def load_or_create_prompt():
    prompt_dir = "prompts"
    prompt_file = os.path.join(prompt_dir, "meeting_summarizer.txt")
    
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)
    
    if not os.path.exists(prompt_file):
        default_prompt = "Please summarize the following transcript:\n\n{transcript}"
        with open(prompt_file, "w") as f:
            f.write(default_prompt)
    
    with open(prompt_file, "r") as f:
        return f.read()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'prompt' not in st.session_state:
    st.session_state.prompt = load_or_create_prompt()

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

def summarize_transcript(client, transcript, prompt):
    summary = get_openai_response(client, prompt.format(transcript=transcript))
    return summary

def save_prompt(prompt):
    prompt_file = os.path.join("prompts", "meeting_summarizer.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)

def save_transcript(transcript):
    transcript_dir = "transcripts"
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)
    
    # Format the timestamp in a more readable format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.txt"
    filepath = os.path.join(transcript_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(transcript)
    
    return filepath

def load_transcript(filepath):
    with open(filepath, "r") as f:
        return f.read()

def main():
    st.title("MeetingMate: Your AI-Powered Meeting Assistant")

    st.markdown("""
    Welcome to MeetingMate! This app helps you:
    1. Record your meetings or voice notes
    2. Automatically transcribe the audio
    3. Generate a concise summary using AI
    
    Perfect for busy professionals who want to save time and capture key points effortlessly.
    """)

    st.header("1. Audio Recording")
    st.info("Click the 'Start recording' button below. The recording will begin when you see the timer start. Click 'Stop recording' when you're done.")
    audio_value = st.experimental_audio_input("Click to record")

    if audio_value is not None:
        st.success("Audio recorded successfully!")
        st.session_state.audio_file = audio_value

        # Automatically transcribe and summarize
        with st.spinner("Transcribing and summarizing..."):
            audio_bytes = st.session_state.audio_file.getvalue()
            
            # Calculate and display file size in MB
            file_size_mb = len(audio_bytes) / (1024 * 1024)
            # st.write(f"Audio file size: {file_size_mb:.2f} MB")
            
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
                audio_file.name = "recording.webm"
                st.session_state.transcript = transcribe_audio(client, audio_file)
            
            # Save the transcript
            st.session_state.transcript_filepath = save_transcript(st.session_state.transcript)
            
            # Display the transcript in an expandable section
            with st.expander("Click to view full transcript", expanded=False):
                st.write(st.session_state.transcript)
            
            # Add download button for transcript
            st.download_button(
                label="Download Transcript",
                data=st.session_state.transcript,
                file_name="meeting_transcript.txt",
                mime="text/plain"
            )

            # Generate and display summary
            st.session_state.summary = summarize_transcript(client, st.session_state.transcript, st.session_state.prompt)
            st.write(st.session_state.summary)

            # Add download button for summary
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name="meeting_summary.txt",
                mime="text/plain"
            )

    # Display and allow editing of the prompt
    st.header("2. Customize Summary Prompt")
    new_prompt = st.text_area("Edit the summary prompt:", st.session_state.prompt, height=100)
    if new_prompt != st.session_state.prompt:
        st.session_state.prompt = new_prompt
        save_prompt(new_prompt)
        st.success("Prompt updated and saved!")

    # Add a button to regenerate summary with the new prompt
    if st.button("Regenerate Summary with New Prompt"):
        if hasattr(st.session_state, 'transcript_filepath') and os.path.exists(st.session_state.transcript_filepath):
            with st.spinner("Regenerating summary..."):
                transcript = load_transcript(st.session_state.transcript_filepath)
                st.session_state.summary = summarize_transcript(client, transcript, st.session_state.prompt)
            st.subheader("Updated Summary:")
            st.write(st.session_state.summary)
        else:
            st.warning("Please record audio before generating a summary.")

if __name__ == "__main__":
    main()