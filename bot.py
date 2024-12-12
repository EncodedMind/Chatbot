from openai import OpenAI
from pathlib import Path
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import time
import warnings
import io
import pydub.playback
from pydub import AudioSegment
from openai import OpenAI

warnings.filterwarnings("ignore", category=DeprecationWarning)

client = OpenAI(
  api_key="YOUR_OPENAI_API_KEY"
)

def record_audio(duration=5, fs=44100, device=None):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, device=device)
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    return fs, audio

def get_chatgpt_response(conversation_history):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {client.api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': "gpt-3.5-turbo",
        'messages': conversation_history,
        'max_tokens': 1000
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        chatgpt_response = response.json()
        return chatgpt_response['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}\nSorry, I couldn't generate a response at the moment. Please try again later."

def main():
    conversation_history = [
        {'role': 'system', 'content': 'You are an empathetic assistant. The user might repeat information. Acknowledge any repetition if necessary, but focus on providing new insights or addressing the user\'s evolving needs. Respond primarily to new information, while taking repeated information into consideration.'}
    ]
    
    while True:
        print("You: ")

        # Record
        duration = 5
        fs = 44100
        device = None
        fs, audio = record_audio(duration, fs, device)
        wav.write("Recording.wav", fs, np.int16(audio * 32767))

        # Transcribe
        with open("Recording.wav", "rb") as audio_file:
            try:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                user_message = transcript.text
            except Exception as e:
                print(f"Error transcribing audio: {e}")
                continue
        
        print(user_message)
        
        # Add user message to conversation history
        conversation_history.append({'role': 'user', 'content': user_message})
        
        response = get_chatgpt_response(conversation_history)

        #Say response out loud
        tts_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=response
        )
        audio_content = io.BytesIO(tts_response.content)
        audio_segment = AudioSegment.from_file(audio_content, format="mp3")
        pydub.playback.play(audio_segment)
        
        # Print and add the response to conversation history
        print("Bot:", response)
        conversation_history.append({'role': 'assistant', 'content': response})

        if user_message.lower() in ("goodbye.", "goodbye!", "goodbye"):
            break

        time.sleep(2)

if __name__ == "__main__":
    tts_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input='Hello! How can I assist you today?'
        )
    audio_content = io.BytesIO(tts_response.content)
    audio_segment = AudioSegment.from_file(audio_content, format="mp3")
    pydub.playback.play(audio_segment)
    main()
