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
    
