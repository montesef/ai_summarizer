# BELOW ARE THE BACKEND FUNCTIONS BEHIND THE APP
import os
from faster_whisper import WhisperModel
import ollama

def transcribe_audio(audio_path):
    """This function will take an audio recording and transcribe it using
    the Faster-Whisper Model that is run locally"""
    model = WhisperModel("small.en",device="auto", compute_type="int8")
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript_text = ""
        for segment in segments:
            transcript_text += segment.text + " "
        return transcript_text
    except Exception as e:
        return f"An error occurred during the transcription: {e}"
    
def generate_summary(transcription):
    # This function will generate a summary of a meeting based on a transcript using the LLM 
    # Ollama, which is run locally
    # BE SURE TO CUSTOMIZE THE PROMPT

    prompt = f"""
    You are an expert executive assistant. Your task is to provide a concise, professional summary of the following meeting transcript.
    Focus on the key decisions made, the main topics discussed, and the overall outcome.
    Give the summary so that anyone may understand what happened in the meeting. Present the summary
    using 200-300 words in a bulleted list.
    Transcript:
    "{transcription}"
    """

    try:
        response = ollama.chat(
            model= "mistral",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error during summarization {e}"
    
def extract_action_items(transcript):
    """
    Extracts action items from a transcript using an OpenAI LLM.
    """
    prompt = f"""
    You are a highly efficient project manager. Your goal is to extract all action items from the following meeting transcript.
    These action items include any tasks that have not been completed during the meeting, meant to be finsihed in the future.
    For each action item, identify the task in detail, the person responsible (Owner), and any mentioned deadline.
    Present the action items in a Markdown table with the columns: 'Task', 'Owner', 'Deadline'.
    If a detail like an Owner or Deadline is not mentioned, write 'Not specified'. The list must have
    enough context so a supervisor can easily understand who is responsible for what task.

    Transcript:
    "{transcript}"
    """
    try:
        response = ollama.chat(
            model= "mistral",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error extracting action items: {e}"
# HERE ENDS THE BACKEND CODE FOR THE APP

import streamlit as st

st.title("🤖 AI Meeting Summarizer")
st.markdown("Upload a meeting audio file (e.g., MP3, M4A) and get a summary and a list of action items.")

# File uploader
uploaded_audio_file = st.file_uploader("Choose an audio file...", type=['mp3', 'mp4', 'm4a', 'wav'])

if uploaded_audio_file is not None:
    st.audio(uploaded_audio_file, format='audio/mp3')

    if st.button("Generate Summary and Action Items"):
        # Save the uploaded file temporarily to pass its path to the functions
        with open(uploaded_audio_file.name, "wb") as f:
            f.write(uploaded_audio_file.getbuffer())

        with st.spinner("Transcribing audio... this may take a moment."):
            transcript_text = transcribe_audio(uploaded_audio_file.name)

        if "Error" not in transcript_text:
            st.header("📄 Meeting Transcript")
            st.text_area("Transcript", transcript_text, height=200)

            with st.spinner("Generating summary..."):
                summary = generate_summary(transcript_text)

            with st.spinner("Extracting action items..."):
                action_items = extract_action_items(transcript_text)

            st.header("📌 Meeting Summary")
            st.markdown(summary)

            st.header("✅ Action Items")
            st.markdown(action_items)
        else:
            st.error(transcript_text) # Show transcription error

        # Clean up the temporary file
        os.remove(uploaded_audio_file.name)