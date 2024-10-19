

Chunk
audio:

Chunking Audio Files with pydub

Overview

This
guide demonstrates how to split large audio files into smaller chunks using the
`pydub` library in Python.

Requirements

Python 3.x

pydub library (`pip install pydub`)

FFmpeg (must be installed and accessible in your system's PATH)

Code Example

```python



from
pydub import AudioSegment



import
io



 



def
split_audio(audio_file, chunk_duration):



    """



    Split an audio file into chunks of
specified duration.



  



    :param audio_file: File-like object or path
to the audio file



    :param chunk_duration: Duration of each
chunk in seconds



    :return: List of AudioSegment objects



    """



    # Load the audio file



    audio = AudioSegment.from_file(audio_file)



  



    # Calculate the total number of chunks



    total_length = len(audio)  # Length in milliseconds



    num_chunks = total_length //
(chunk_duration * 1000) + (total_length % (chunk_duration * 1000) > 0)



 



    chunks = []



    # Split into chunks



    for i in range(num_chunks):



        start_time = i * chunk_duration * 1000



        end_time = start_time + chunk_duration
* 1000



        chunk = audio[start_time:end_time]



        chunks.append(chunk)



  



    return chunks



 



#
Example usage



CHUNK_DURATION
= 300  # 5 minutes in seconds



 



#
For file path



chunks
= split_audio("path/to/your/audio.mp3", CHUNK_DURATION)



 



#
For file-like object (e.g., from Streamlit's file_uploader)



uploaded_file
= ... # Your file object



chunks
= split_audio(uploaded_file, CHUNK_DURATION)



 



#
Process chunks (e.g., convert to bytes for API submission)



for
chunk in chunks:



    buffer = io.BytesIO()



    chunk.export(buffer,
format="mp3")



    audio_bytes = buffer.getvalue()



    # Use audio_bytes with your API or further
processing



```

Notes

This method works with various audio formats (MP3, WAV, etc.) as long as FFmpeg
supports them.

Adjust `CHUNK_DURATION` based on your specific needs or API limitations.

The `split_audio` function returns a list of `AudioSegment` objects, which can
be further processed or exported as needed.

When working with file-like objects (e.g., from Streamlit's `file_uploader`),
pydub can read directly from the object.

To convert chunks to bytes for API submission, use the `export` method with an
`io.BytesIO` buffer.


````markdown
# Audio Capabilities with Gemini API

## Overview
The Gemini API processes audio inputs to describe, summarize, transcribe, or answer questions about audio content.

## Supported Audio Formats
WAV, MP3, AIFF, AAC, OGG Vorbis, FLAC

## Technical Details
- 1 second of audio = 25 tokens
- Maximum audio length: 9.5 hours
- Downsampled to 16 Kbps, multiple channels combined

## Usage

### 1. Set up project and API key
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
```

### 2. Upload and process audio file
```python
# Upload file
myfile = genai.upload_file("path/to/audio.mp3")

# Generate content
model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content([myfile, "Describe this audio clip"])
print(result.text)
```

### 3. Inline audio processing
```python
model = genai.GenerativeModel('models/gemini-1.5-flash')
response = model.generate_content([
    "Summarize the audio.",
    {
        "mime_type": "audio/mp3",
        "data": pathlib.Path('audio.mp3').read_bytes()
    }
])
print(response.text)
```

## Key Functions
- `upload_file()`: Upload audio file
- `generate_content()`: Process audio and generate response
- `get_file()`: Retrieve file metadata
- `list_files()`: List uploaded files
- `delete()`: Manually delete uploaded file

## Additional Features

### Transcription
```python
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content([
    "Generate a transcript of the speech.",
    audio_file
])
print(response.text)
```

### Timestamp-specific analysis
```python
prompt = "Provide a transcript from 02:30 to 03:29."
response = model.generate_content([prompt, audio_file])
print(response.text)
```

### Token counting
```python
token_count = model.count_tokens([audio_file])
print(token_count)
```

## Limitations
- English-language speech only
- Cannot generate audio output
````

This single block contains the entire concise documentation for the Gemini API's audio capabilities, including an overview, supported formats, technical details, usage instructions with code samples, key functions, additional features, and limitations.
