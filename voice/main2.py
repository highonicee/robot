from audioinput import record_audio
from speechmodule import speech_to_text, speak
from intentmodule import classify_query
from news import get_news
from wiki import WikiAssistant
from llamamodule import ask_llm
from wakeword import detect_wake_word

wiki_assistant = WikiAssistant()
def process_query(query):

    # 🔴 EMPTY INPUT BLOCKER (MOST IMPORTANT FIX)
    if not query or query.strip() == "":
        return "I didn't hear anything. Please say it again."

    intent = classify_query(query)
    print(f"[DEBUG] Intent detected: {intent}")

    query = query.lower()

    # -------------------------
    # CASUAL CONVERSATION
    # -------------------------
    casual_words = ["hello", "hi", "how are you", "what's up", "hey"]

    if any(word in query for word in casual_words):
        return ask_llm(query)

    # -------------------------
    # NEWS
    # -------------------------
    if intent == "news":
        return get_news(query)

    # -------------------------
    # FACTUAL → WIKI + FALLBACK
    # -------------------------
    elif intent == "factual":

        result = wiki_assistant.get_summary(query)

        # 🔴 HANDLE ALL FAILURE CASES
        if result in ["EMPTY_QUERY"]:
            return "I didn't catch that properly."

        if result in ["NO_RESULT", "WIKI_ERROR"]:
            print("[DEBUG] Wiki failed → using LLM")
            return ask_llm(f"Answer this factually:\n{query}")

        return wiki_assistant.format_for_speech(result)

    # -------------------------
    # REASONING
    # -------------------------
    elif intent == "reasoning":
        return ask_llm(query)

    # -------------------------
    # DEFAULT
    # -------------------------
    else:
        return ask_llm(query)

# MAIN LOOP
def main():

    print("=== Voice Assistant Started ===")
    print("Say 'exit' to stop\n")

    while True:
        detect_wake_word()

        speak("Yes, how can I help you?")

        wav_file = record_audio(silence_threshold=0.01, silence_duration=1.8)

        query = speech_to_text(wav_file)

        # 🔴 ADD THIS HERE (INSIDE LOOP)
        if not query or query.strip() == "":
            speak("I didn't hear anything.")
            continue

        response = process_query(query)

        speak(response)


# -------------------------------
if __name__ == "__main__":
    main()