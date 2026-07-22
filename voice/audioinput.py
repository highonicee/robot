"""
voice/audioinput.py
===================
Fixed version:
- speech_started threshold lowered to match actual mic levels (0.010)
- Max recording time reduced to 8s (was 15s — caused the 15s silence bug)
- Minimum speech duration added so robot doesn't respond to a cough
- Device index 8 hardcoded (Realtek, confirmed working)
"""

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

DEVICE_INDEX     = 8
SILENCE_THRESHOLD = 0.010   # just above noise floor 0.01379 from test
FRAMES_PER_BLOCK  = 1024


def record_audio(
    filename: str = None,
    fs: int = 16000,
    silence_threshold: float = SILENCE_THRESHOLD,
    silence_duration: float = 1.5,   # stop after 1.5s silence (was 2.0)
    min_speech_duration: float = 0.3 # ignore clips shorter than 0.3s
) -> str:
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "input.wav")

    print(f"[Audio] Listening... (threshold={silence_threshold:.4f})")

    recording        = []
    silence_counter  = 0
    speech_frames    = 0
    speech_started   = False
    silence_limit    = int(silence_duration * fs / FRAMES_PER_BLOCK)
    min_frames       = int(min_speech_duration * fs / FRAMES_PER_BLOCK)

    def callback(indata, frames, time_info, status):
        nonlocal silence_counter, speech_started, speech_frames

        # indata is int16 from InputStream — normalize to float for RMS
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)) / 32768.0)
        recording.append(indata.copy())

        if rms > silence_threshold:
            speech_started  = True
            speech_frames  += 1
            silence_counter = 0
        elif speech_started:
            silence_counter += 1

        if speech_started and silence_counter > silence_limit:
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype='int16',
            blocksize=FRAMES_PER_BLOCK,
            device=DEVICE_INDEX,
            callback=callback
        ):
            sd.sleep(8000)   # max 8s — was 15s causing the long silence bug
    except sd.CallbackStop:
        pass

    if not recording or speech_frames < min_frames:
        print("[Audio] No speech detected.")
        wav.write(filename, fs, np.zeros(fs, dtype=np.int16))
        return filename

    audio = np.concatenate(recording, axis=0)
    audio = np.squeeze(audio)
    wav.write(filename, fs, audio)
    print(f"[Audio] Recorded {len(audio)/fs:.1f}s")
    return filename