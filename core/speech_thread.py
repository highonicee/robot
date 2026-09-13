"""
core/speech_thread.py
=====================
FIXES applied:
  FIX 1: Clean terminal — separator lines around speech events so they
          don't get lost in YOLO output.
  FIX 3: Wake-once session model.
          - Wake word activates a SESSION.
          - Within the session the robot answers as many queries as you ask.
          - Session ends only when the user says a shutdown phrase
            ("goodbye", "bye", "shutdown", "stop listening", "go to sleep").
          - After session ends, robot goes back to listening for wake word.
  FIX 4: Vision routing (depends on fixed intent_router.py).
"""

import sys
import os
import threading
import time

_CORE_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_CORE_DIR, ".."))
_VOICE_DIR  = os.path.join(_ROOT, "voice")
_ROBOT_FACE = os.path.join(_ROOT, "robot_face")
_VISION_DIR = os.path.join(_ROBOT_FACE, "vision")

for _p in [_ROOT, _VOICE_DIR, _ROBOT_FACE, _VISION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.memory        import memory
from core.event_bus     import bus, Events
from core.intent_router import classify_query
from core.vision_query  import answer_vision_query
import core.vision_thread as vt_module   # for STT busy flag

from voice.audioinput   import record_audio
from voice.speechmodule import speech_to_text, speak
from voice.news         import get_news
from voice.wiki         import WikiAssistant
from voice.llamamodule  import ask_llm
from voice.wakeword     import detect_wake_word

# Whisper hallucination phrases — ignore these
_HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you.",
    "thanks for watching", "bye.", "what is science",
    "what is science?", "what is science? what is science?",
    "please subscribe", "see you next time",
}

# FIX 3: Phrases that end the active session (robot goes back to sleep)
_SHUTDOWN_SESSION_PHRASES = [
    "goodbye", "good bye", "bye", "see you", "see you later",
    "shutdown", "shut down", "go to sleep", "stop listening",
    "stop", "exit", "turn off", "sleep",
]

def _is_session_end(query: str) -> bool:
    q = query.lower().strip().rstrip(".")
    return any(phrase in q for phrase in _SHUTDOWN_SESSION_PHRASES)


class SpeechThread:

    def __init__(self):
        self._thread        = None
        self._stop_event    = threading.Event()
        self._ready_event   = threading.Event()
        self._speaking_lock = threading.Lock()
        self._wiki          = WikiAssistant()

        bus.subscribe(Events.FACE_RECOGNIZED, self._on_face_recognized)
        bus.subscribe(Events.SHUTDOWN,        self._on_shutdown)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="SpeechThread", daemon=True
        )
        self._thread.start()
        print("[Speech] Thread started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[Speech] Thread stopped.")

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    # ──────────────────────────────────────────────────────────────
    # MAIN LOOP  (FIX 3: wake-once, session-based)
    # ──────────────────────────────────────────────────────────────
    def _run(self):
        self._ready_event.set()
        bus.publish(Events.SPEECH_READY)

        self._log("Ready. Waiting for wake word...")

        while not self._stop_event.is_set():
            try:
                # ── Phase 1: wait for wake word (blocks here) ────────────
                detect_wake_word()

                if self._stop_event.is_set():
                    break

                self._safe_speak("Yes?")
                self._log("Session started. Listening for queries.")
                self._log("(Say 'goodbye' or 'shutdown' to end the session)")

                # ── Phase 2: active session ───────────────────────────────
                # Robot keeps answering until it hears a goodbye phrase.
                while not self._stop_event.is_set():

                    # Record query
                    self._log("Listening...")
                    wav_file = record_audio(
                        silence_threshold=0.010,
                        silence_duration=1.5
                    )

                    # Transcribe — notify vision to reduce CPU load
                    self._log("Transcribing...")
                    vt_module.notify_stt_start()
                    try:
                        query = speech_to_text(wav_file)
                    finally:
                        vt_module.notify_stt_end()

                    # Validate — reject empty / hallucinated text
                    if not query or query.strip().lower() in _HALLUCINATIONS:
                        self._log(f"Rejected: '{query}' (empty or hallucination)")
                        self._safe_speak("Sorry, I didn't catch that.")
                        continue

                    self._log(f"Query: '{query}'")
                    bus.publish(Events.QUERY_RECEIVED, query=query)

                    # FIX 3: check for session-end phrase
                    if _is_session_end(query):
                        self._safe_speak(
                            "Okay, going back to sleep. Say 'Dora' to wake me up again."
                        )
                        self._log("Session ended by user.")
                        break   # break inner loop → outer loop waits for wake word again

                    # Process and respond
                    response = self._process_query(query)
                    self._log(f"Response: {response}")
                    self._safe_speak(response)

            except Exception as e:
                print(f"[Speech] Error: {e}")
                time.sleep(1)

    # ──────────────────────────────────────────────────────────────
    # QUERY PROCESSING
    # ──────────────────────────────────────────────────────────────
    def _process_query(self, query: str) -> str:
        intent = classify_query(query)
        self._log(f"Intent: {intent}")

        q = query.lower()

        # Casual conversation
        casual = ["hello", "hi", "how are you", "what's up", "hey"]
        if any(w in q for w in casual):
            return ask_llm(query)

        # FIX 4: Vision queries route to shared memory
        if intent == "vision":
            answer = answer_vision_query(query)
            return answer if answer else "I'm not sure what to look for."

        if intent == "news":
            return get_news(query)

        if intent == "factual":
            result = self._wiki.get_summary(query)
            if result in ("EMPTY_QUERY",):
                return "I didn't catch that properly."
            if result in ("NO_RESULT", "WIKI_ERROR"):
                return ask_llm(f"Answer this factually:\n{query}")
            return self._wiki.format_for_speech(result)

        return ask_llm(query)

    # ──────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ──────────────────────────────────────────────────────────────
    def _on_face_recognized(self, name, emotion, action, bbox):
        if not memory.should_greet(name):
            return
        memory.mark_greeted(name)
        emotion = (emotion or "neutral").lower()
        if emotion in ("happy", "surprise"):
            mood = "You're looking great!"
        elif emotion in ("sad", "fear"):
            mood = "Hope you're doing okay."
        elif emotion in ("angry", "disgust"):
            mood = "Hope your day gets better."
        else:
            mood = "Good to see you!"
        self._safe_speak(f"Hi {name}! {mood}")

    def _on_shutdown(self):
        self._stop_event.set()

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────
    def _safe_speak(self, text: str):
        with self._speaking_lock:
            bus.publish(Events.ROBOT_SPEAKING, text=text)
            speak(text)

    def _log(self, msg: str):
        """
        FIX 1: Print speech events with a clear visual separator so they
        stand out even when YOLO lines appear in between.
        """
        print(f"\n>>> [Speech] {msg}")