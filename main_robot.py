"""
main_robot.py
=============
Mini Dora — Unified Entry Point
================================
Starts both vision and speech threads together.
Vision writes to shared memory. Speech reads from it.

Run with:
    python main_robot.py

Requirements:
    - venv  (vision deps) must be active  OR
    - venv_voice (speech deps) must be active
    → See README_INTEGRATION.md for how to run both envs together.
"""

import sys
import os
import time
import signal
import threading

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR   = os.path.join(ROOT, "voice")                    # ROBOT/voice/
ROBOT_FACE  = os.path.join(ROOT, "robot_face")               # ROBOT/robot_face/
VISION_DIR  = os.path.join(ROBOT_FACE, "vision")             # ROBOT/robot_face/vision/

for _p in [ROOT, VOICE_DIR, ROBOT_FACE, VISION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Core modules ──────────────────────────────────────────────────────────────
from core.memory import memory
from core.event_bus import bus, Events
from core.vision_thread import VisionThread
from core.speech_thread import SpeechThread

# STARTUP BANNER
def print_banner():
    print("""
╔══════════════════════════════════════════════╗
║           🤖  Mini Dora  —  v1.0             ║
║   Integrated Vision + Speech Robot System    ║
╚══════════════════════════════════════════════╝
    """)

# SHUTDOWN HANDLER
_shutdown_event = threading.Event()

def _handle_signal(sig, frame):
    print("\n[Main] Shutdown signal received. Stopping...")
    _shutdown_event.set()

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# MAIN
def main():
    print_banner()

    # ── Init threads ──────────────────────────────────────────────────────────
    vision_thread = VisionThread(camera_index=0, show_window=True)
    speech_thread = SpeechThread()

    # ── Start vision first (models take time to load) ─────────────────────────
    print("[Main] Starting vision system...")
    vision_thread.start()

    print("[Main] Waiting for vision to initialize...")
    if not vision_thread.wait_until_ready(timeout=20.0):
        print("[Main] WARNING: Vision did not initialize in time. Continuing anyway.")

    # ── Start speech ──────────────────────────────────────────────────────────
    print("[Main] Starting speech system...")
    speech_thread.start()
    speech_thread.wait_until_ready(timeout=10.0)

    # ── Startup greeting ──────────────────────────────────────────────────────
    time.sleep(1.0)  # let TTS engine settle
    from voice.speechmodule import speak
    speak("Hello, my name is Mini Dora. I am ready to help you.")

    print("[Main] Both systems running. Press Ctrl+C to stop.")
    print("[Main] Say a wake word to interact (e.g. 'Hey Mini Dora')\n")

    # ── Keep main thread alive until shutdown ─────────────────────────────────
    while not _shutdown_event.is_set():
        time.sleep(0.5)

    # ── Graceful shutdown ─────────────────────────────────────────
    print("[Main] Shutting down...")

    from voice.speechmodule import speak   # import here (safe)

    speak("Mini Dora offline. Goodbye.")

    bus.publish(Events.SHUTDOWN)

    vision_thread.stop()
    speech_thread.stop()

    print("[Main] Mini Dora   offline.   Goodbye.")


if __name__ == "__main__":
    main()