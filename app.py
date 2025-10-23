# BELOW ARE THE BACKEND FUNCTIONS BEHIND THE APP
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_path):
    """This function will take an audio recording and transcribe it using
    OpenAI's Whisper Model"""
    try:
        with open(audio_path,"rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model = "whisper-1",
                file = audio_file
            )
            return transcript.text
    except Exception as e:
        return f"An error occurred during the transcription: {e}"
    
def generate_summary(transcription):
    # This function will generate a summary of a meeting based on a transcript using gpt-4o
    # BE SURE TO CUSTOMIZE THE PROMPT

    prompt = f"""
    You are an expert executive assistant. Your task is to provide a concise, professional summary of the following meeting transcript.
    Focus on the key decisions made, the main topics discussed, and the overall outcome.
    Present the summary in three or more clear bullet points.

    Transcript:
    "{transcription}"
    """

    try:
        response = openai.chat.completions.create(
            model = 'gpt-4o',
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during summarization {e}"
    
def extract_action_items(transcript):
    """
    Extracts action items from a transcript using an OpenAI LLM.
    """
    prompt = f"""
    You are a highly efficient project manager. Your goal is to extract all action items from the following meeting transcript.
    For each action item, identify the task, the person responsible (Owner), and any mentioned deadline.
    Present the action items in a Markdown table with the columns: 'Task', 'Owner', 'Deadline'.
    If a detail like an Owner or Deadline is not mentioned, write 'Not specified'.

    Transcript:
    "{transcript}"
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in structured data extraction."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error extracting action items: {e}"
# HERE ENDS THE BACKEND CODE FOR THE APP

import streamlit