"""
voice/speechmodule.py
=====================
Uses Whisper 'base' (not medium) — medium is 4x slower on CPU,
causes vision thread spikes of 300-1000ms seen in the logs.
Initial prompt tuned for Indian English + robot commands.
"""

import whisper
import pyttsx3
import time

# base = 74M params, ~3-5s on CPU — acceptable
# medium = 769M params, ~30-60s on CPU — too slow, freezes vision
model = whisper.load_model("base")

WHISPER_PROMPT = (
    "Indian English speaker. Robot voice commands. "
    "Short questions about objects, people, colors, locations."
)


def speech_to_text(wav_file: str) -> str:
    result = model.transcribe(
        wav_file,
        language="en",
        initial_prompt=WHISPER_PROMPT,
        temperature=0.0,
        best_of=1,
        beam_size=3,       # reduced from 5 — faster on CPU
        condition_on_previous_text=False,
        fp16=False,
    )
    text = result["text"].strip()
    return text.lower()


def speak(text: str):
    print(f"[Robot] Speaking: {text}")
    try:
        engine = pyttsx3.init(driverName='sapi5')
        engine.setProperty('rate', 155)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS] Error: {e}")
    time.sleep(0.2)