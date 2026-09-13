"""
wakeword_test.py  — fixed version
Place in ROBOT/voice/ and run: python wakeword_test.py
"""

import sounddevice as sd
import numpy as np
import queue
import json
import os
import sys
import time

try:
    from vosk import Model, KaldiRecognizer
    print("[OK] Vosk imported")
except ImportError:
    print("[ERROR] pip install vosk")
    sys.exit(1)

# ─────────────────────────────────────────────
# STEP 1 — Find model
# ─────────────────────────────────────────────
ROBOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_SEARCH_PATHS = [
    os.path.join(ROBOT_ROOT, "data", "vosk-model-en-in-0.5"),
    os.path.join(ROBOT_ROOT, "data", "vosk-model-small-en-in-0.4"),
    os.path.join(ROBOT_ROOT, "data", "vosk_model"),
    os.path.join(ROBOT_ROOT, "data", "vosk-model-small-en-us-0.15"),
    os.path.join(os.path.dirname(__file__), "..", "data", "vosk_model"),
]

print("\n── Step 1: Looking for Vosk model ──")
model_path = None
for p in MODEL_SEARCH_PATHS:
    norm = os.path.normpath(p)
    exists = os.path.exists(norm)
    print(f"  {'[FOUND]' if exists else '[ -- ]'}  {norm}")
    if exists and model_path is None:
        model_path = norm

if not model_path:
    print("\n[ERROR] No model found.")
    print("  Download: https://alphacephei.com/vosk/models")
    print("  Get: vosk-model-small-en-in-0.4  → extract to ROBOT/data/")
    sys.exit(1)

print(f"\n[OK] Model: {os.path.basename(model_path)}")

# ─────────────────────────────────────────────
# STEP 2 — List devices + pick best mic
# ─────────────────────────────────────────────
print("\n── Step 2: Audio devices ────────────")
devices = sd.query_devices()
input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

for i, d in input_devices:
    marker = " ← default" if i == sd.default.device[0] else ""
    print(f"  [{i:2d}] {d['name'][:50]:<50} {int(d['default_samplerate'])} Hz{marker}")

# Try to find a better mic than webcam
DEFAULT_MIC = sd.default.device[0]
PREFERRED_MIC = DEFAULT_MIC

# Prefer Realtek or headset over webcam
for i, d in input_devices:
    name = d['name'].lower()
    if 'realtek' in name and 'audio' in name:
        PREFERRED_MIC = i
        break
    if 'headset' in name or 'headphone' in name:
        PREFERRED_MIC = i
        break

if PREFERRED_MIC != DEFAULT_MIC:
    print(f"\n  [HINT] Webcam mic detected as default.")
    print(f"  Switching to device [{PREFERRED_MIC}]: {devices[PREFERRED_MIC]['name']}")
    print(f"  (Set DEVICE_INDEX below to override manually)")
else:
    print(f"\n  Using default device [{DEFAULT_MIC}]: {devices[DEFAULT_MIC]['name']}")

# ── SET THIS MANUALLY IF WRONG ─────────────────────────────────────────
# Look at the list above and change to the device index you want.
# e.g. DEVICE_INDEX = 8  for Realtek(R) Audio at 44100 Hz
DEVICE_INDEX = 8
# ───────────────────────────────────────────────────────────────────────

print(f"\n  >>> Using device [{DEVICE_INDEX}]: {devices[DEVICE_INDEX]['name']}")
print(f"      To change: edit DEVICE_INDEX at line ~70 of this file\n")

# ─────────────────────────────────────────────
# STEP 3 — Volume check with chosen device
# ─────────────────────────────────────────────
print("── Step 3: Mic volume check ─────────")
print("  Quiet for 1.5s (noise floor)...")

try:
    sil = sd.rec(
        int(1.5 * 16000), samplerate=16000, channels=1,
        dtype='float32', device=DEVICE_INDEX
    )
    sd.wait()
    noise_rms = float(np.sqrt(np.mean(sil ** 2)))
    print(f"  Noise floor: {noise_rms:.5f}")
except Exception as e:
    print(f"  [ERROR] Could not record from device {DEVICE_INDEX}: {e}")
    print("  Try changing DEVICE_INDEX to a different mic number above.")
    sys.exit(1)

print("  Speak normally for 2s — say 'hello dora'...")
time.sleep(0.3)
try:
    speech = sd.rec(
        int(2 * 16000), samplerate=16000, channels=1,
        dtype='float32', device=DEVICE_INDEX
    )
    sd.wait()
    speech_rms = float(np.sqrt(np.mean(speech ** 2)))
    print(f"  Speech RMS:  {speech_rms:.5f}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

ratio = speech_rms / noise_rms if noise_rms > 0 else 0
if ratio < 2:
    print(f"\n  [WARNING] Ratio {ratio:.1f}x — mic signal weak!")
    print("  → Speak louder, move closer, or change DEVICE_INDEX")
else:
    print(f"\n  [OK] Ratio {ratio:.1f}x — mic signal good")

dynamic_threshold = max(noise_rms * 2.0, 0.003)
print(f"  Dynamic threshold will be: {dynamic_threshold:.5f}")

# ─────────────────────────────────────────────
# STEP 4 — Load model
# ─────────────────────────────────────────────
print("\n── Step 4: Loading Vosk model ───────")
t0 = time.time()
try:
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    print(f"  [OK] Loaded in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# STEP 5 — Live recognition
# ─────────────────────────────────────────────
WAKE_WORDS = ["dora", "mini dora", "hey dora", "hello dora", "ok dora", "hi dora"]
FUZZY      = ["door a", "minnie", "many dora", "me dora", "mini door",
              "many door", "robot", "mendora", "nidora", "mirror"]

print("\n── Step 5: LIVE recognition ─────────")
print("  Every word Vosk hears will print below.")
print("  Wake words:", WAKE_WORDS)
print("  Press Ctrl+C to stop.\n")
print("─" * 55)

q_audio = queue.Queue()
last_bar = time.time()

def callback(indata, frames, time_info, status):
    # ── FIX: RawInputStream gives a cffi buffer — convert to bytes only ──
    # Do NOT call .copy() on cffi buffer objects
    if status:
        sys.stderr.write(f"[audio] {status}\n")
    q_audio.put(bytes(indata))

try:
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype='int16',
        channels=1,
        device=DEVICE_INDEX,
        callback=callback
    ):
        print(f"  Mic [{DEVICE_INDEX}] open. Say something!\n")

        while True:
            raw = q_audio.get()   # plain bytes — no .copy() needed

            # ── Volume bar ────────────────────────────────────────────────
            now = time.time()
            if now - last_bar > 0.35:
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(pcm ** 2)) / 32768.0
                filled = int(min(rms / dynamic_threshold * 20, 40))
                bar = "█" * filled + "░" * (40 - filled)
                thresh_marker = int(20)  # threshold line at middle
                level = "LOUD" if rms > dynamic_threshold else "quiet"
                print(f"  [{bar}] {rms:.4f} {level}   ", end="\r")
                last_bar = now

            # ── Partial (real-time) ───────────────────────────────────────
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial:
                print(f"\n  [PARTIAL] {partial}")
                pl = partial.lower()
                for w in WAKE_WORDS + FUZZY:
                    if w in pl:
                        print(f"  >>> WAKE MATCH (partial): '{w}'  ✅\n")

            # ── Final (after silence) ─────────────────────────────────────
            if rec.AcceptWaveform(raw):
                text = json.loads(rec.Result()).get("text", "").strip()
                if text:
                    tl = text.lower()
                    print(f"\n  [FINAL ] '{text}'")
                    matched = False
                    for w in WAKE_WORDS:
                        if w in tl:
                            print(f"  >>> WAKE WORD MATCHED: '{w}'  ✅")
                            matched = True
                    for w in FUZZY:
                        if w in tl:
                            print(f"  >>> FUZZY MATCH: '{w}'  — add to WAKE_WORDS  ✅")
                            matched = True
                    if not matched:
                        print(f"  ✗  No match. Vosk heard: '{text}'")
                        print(f"     → Add '{text}' to FUZZY list if that's what you said")
                    print()

except KeyboardInterrupt:
    print("\n\n── Diagnosis summary ────────────────")
    print("  VOL bar always empty → mic wrong device, change DEVICE_INDEX")
    print("  VOL moves, no PARTIAL → Vosk not recognising — get Indian model")
    print("  PARTIAL shows wrong words → copy them into FUZZY list")
    print("  WAKE MATCH fires → it works, use that word in wakeword.py")
