import speech_recognition as sr
import numpy as np

def listen_and_recognize():
    """
    Captures audio using the system's default microphone via sounddevice
    and converts it to text using Google's Speech API.
    """
    recognizer = sr.Recognizer()
    
    # We use Microphone() but since sounddevice is installed, 
    # SpeechRecognition will use it as a fallback if PyAudio is missing.
    with sr.Microphone() as source:
        print("\n[LISTENING] Speak now...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        print("[RECOGNIZING] Processing your voice...")
        query = recognizer.recognize_google(audio)
        print(f"[USER] {query}")
        return query.lower()
    except sr.UnknownValueError:
        return "None"
    except sr.RequestError:
        print("[ERROR] Internet connection issue for Speech API.")
        return "None"