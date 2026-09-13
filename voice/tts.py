import pyttsx3

def speak(text):
    """
    Converts text to audible speech.
    """
    engine = pyttsx3.init()
    # Fine-tuning for a humanoid voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id) # 0 for male, 1 for female
    engine.setProperty('rate', 180) # WPM (Words Per Minute)
    
    print(f"[ROBOT] {text}")
    engine.say(text)
    engine.runAndWait()