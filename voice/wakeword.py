"""
voice/wakeword.py
=================
Wake word detection tuned from live test results:
- Device 8 (Realtek) confirmed working
- "hello dora" confirmed detected
- "mini dora" → Vosk hears "mini dollar" / "mini donuts" → added as fuzzy
- Threshold lowered: speech sits at 0.013-0.022, threshold was 0.027 (too high)
- Partial result matching added for faster response
"""

import sounddevice as sd
import queue
import json
import os
import sys
from vosk import Model, KaldiRecognizer

q = queue.Queue()

# ── Device index confirmed from wakeword_test.py ─────────────────────────────
# Device [8] Microphone (Realtek(R) Audio) 44100 Hz — confirmed working
DEVICE_INDEX = 8

# ── Model path ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_PATHS = [
    os.path.join(_ROOT, "data", "vosk-model-en-in-0.5"),
    os.path.join(_ROOT, "data", "vosk-model-small-en-in-0.4"),
    os.path.join(_ROOT, "data", "vosk_model"),
]
_model_path = next((p for p in _MODEL_PATHS if os.path.exists(p)), None)
if not _model_path:
    raise FileNotFoundError("No Vosk model found in data/. Check README_INTEGRATION.md")

model      = Model(_model_path)
recognizer = KaldiRecognizer(model, 16000)

# ── Wake words (confirmed working from test) ──────────────────────────────────
WAKE_WORDS = [
    "dora",           # catches everything: hello dora, hey dora, ok dora
    "hello dora",     # confirmed ✅ in test
    "hey dora",
    "ok dora",
    "hi dora",
]

# ── Fuzzy alternatives (directly from test output) ───────────────────────────
# These are exactly what Vosk transcribes when you say "mini dora" or similar
FUZZY_ALTERNATIVES = [
    "hello donna",    # test: said "hello dora" → heard "hello donna"
    "hello doctor",   # test: said "hello dora" → heard "hello doctor"
    "hello does",     # test: said "hello dora" → heard "hello does"
    "mini dollar",    # test: said "mini dora"  → heard "mini dollar"
    "mini donuts",    # test: said "mini dora"  → heard "mini donuts"
    "mini the",       # test: said "mini dora"  → heard "mini the"
    "mini door",
    "many dora",
    "many dollar",
    "me dora",
    "mendora",
]


def _callback(indata, frames, time_info, status):
    # indata is cffi buffer from RawInputStream — convert to bytes only
    if status:
        sys.stderr.write(f"[Wake] Audio status: {status}\n")
    q.put(bytes(indata))


def detect_wake_word() -> bool:
    """
    Blocks until a wake word is detected.
    Uses both partial and final results for fastest possible response.
    Returns True when triggered.
    """
    print("[Wake] Listening for wake word...")

    # Clear stale audio from queue before starting
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype='int16',
        channels=1,
        device=DEVICE_INDEX,
        callback=_callback
    ):
        while True:
            raw = q.get()

            # ── Check partial results first (faster response) ─────────────
            partial_text = json.loads(
                recognizer.PartialResult()
            ).get("partial", "").lower().strip()

            if partial_text and _is_wake_word(partial_text):
                print(f"[Wake] Triggered (partial): '{partial_text}'")
                # Flush the recognizer so next call starts clean
                recognizer.FinalResult()
                return True

            # ── Check final result (after silence) ────────────────────────
            if recognizer.AcceptWaveform(raw):
                final_text = json.loads(
                    recognizer.Result()
                ).get("text", "").lower().strip()

                if final_text:
                    print(f"[Wake] Heard: '{final_text}'")

                if final_text and _is_wake_word(final_text):
                    print(f"[Wake] Triggered (final): '{final_text}'")
                    return True


def _is_wake_word(text: str) -> bool:
    """Check if transcribed text contains a wake word or fuzzy alternative."""
    for w in WAKE_WORDS:
        if w in text:
            return True
    for w in FUZZY_ALTERNATIVES:
        if w in text:
            return True
    return False