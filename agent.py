import io
import json
import requests
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
import client as socket_client  # Renamed so OpenAI doesn't overwrite your socket file
import time
import datetime
import readline
import os
base_dir = os.getcwd()
doc_dir = os.path.expanduser("~/Documents")
downloads_dir = os.path.expanduser("~/Downloads")
print("Current:", base_dir)
print("Documents:", doc_dir)
print("Downloads:", downloads_dir)

ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
INSTRUCTIONS = "You are Ben, a wise, calm and responsible AI assistant. Keep replies short since they are spoken aloud."
os.makedirs("chat_logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
chat_file = f"chat_logs/chat_{timestamp}.jsonl"
conversation_history = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]
def save_message(role, content):
    entry = {"role": role, "content": content}
    conversation_history.append(entry)

    with open(chat_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

print(f"Started new chat: {chat_file}\n")
while True:
   
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    save_message("user", user_input)
    ai_reply = f"Echoing your message (history has {len(conversation_history)} items)"
    
    print(f"AI: {ai_reply}\n")
    save_message("assistant", ai_reply)
def speak(text):
    response = ai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
        instructions="Speak like a wise, calm and responsible AI Assistant named Ben.",
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
        dtype="int16",
    )
    sd.wait()
    print("Processing...")

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, recording, samplerate, format="WAV")
    audio_buffer.seek(0)
    audio_buffer.name = "audio.wav"

    transcript = ai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
    )

    print(f"You said: {transcript.text}")
    return transcript.text

def get_weather(city):
    try:
        geo = requests.get(
            "https://open-meteo.com",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Couldn't find a place called {city}."
        place = geo["results"][0]

        w = requests.get(
            "https://open-meteo.com",
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


def ask_ai(text, max_steps=5):
    input_items = [{"role": "user", "content": text}]

    for _ in range(max_steps):
        response = ai_client.responses.create(
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


if __name__ == "__main__":
    while True:
        user_text = listen()
        if not user_text.strip():
            continue
        if "goodbye" in user_text.lower() or "exit" in user_text.lower():
            speak("Goodbye.")
            break
        reply = ask_ai(user_text)
        print(f"Ben: {reply}")
        speak(reply)

