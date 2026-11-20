
import client
import os
import webbrowser
import time
import random
import requests
import io
import soundfile as sf
import numpy as np
import cv2
from datetime import datetime
from openai import OpenAI
import sounddevice as sd


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 
def speak(text):
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
        instructions="Speak like a wise, calm and responsible AI Assistant named Ben."
    )
    audio_bytes = io.BytesIO(response.content)
    data, samplerate = sf.read(audio_bytes)
    sd.play(data, samplerate)
    sd.wait()

def listen(duration=5, samplerate=44100):
    print("Listening...")
    recording = sd.rec(
        int(duration * samplerate), 
        samplerate=samplerate,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    print("Processing...")

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, recording, samplerate, format="WAV")
    audio_buffer.seek(0)
    audio_buffer.name = "audio.wav"

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer
    )
    

    print(f"You said: {transcript.text}")
    return transcript.text

def ask_ai(text):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=text
    )
    output = response.output_text
    return output


#weather function
import json
import requests

def get_weather(city):
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Couldn't find a place called {city}."
        place = geo["results"][0]

        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,wind_speed_10m,precipitation",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
            },
            timeout=10,
        ).json()["current"]

        return (
            f"{place['name']}: {w['temperature_2m']}F "
            f"(feels like {w['apparent_temperature']}F), "
            f"wind {w['wind_speed_10m']} mph, precipitation {w['precipitation']} mm."
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
    },
]

TOOL_FUNCS = {
    "get_weather": get_weather,
}

INSTRUCTIONS = "You are Ben, a wise, calm and responsible AI assistant. Keep replies short since they are spoken aloud."


def ask_ai(text, max_steps=5):
    input_items = [{"role": "user", "content": text}]

    for _ in range(max_steps):
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=INSTRUCTIONS,
            input=input_items,
            tools=TOOL_SCHEMAS,
        )
        input_items += response.output

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        for call in calls:
            fn = TOOL_FUNCS.get(call.name)
            try:
                args = json.loads(call.arguments or "{}")
                result = fn(**args) if fn else f"Unknown tool: {call.name}"
            except Exception as e:
                result = f"Tool error: {e}"
            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": str(result),
            })

    return "I got stuck using my tools, sorry."


