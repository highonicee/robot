import sounddevice as sd
import numpy as np

# 🔥 SET YOUR DEVICES HERE
INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)

samplerate = 16000
duration = 5

print("Recording... Speak now!")

audio = sd.rec(int(duration * samplerate),
               samplerate=samplerate,
               channels=1,
               dtype='int16')

sd.wait()

print("Playing back...")

sd.play(audio, samplerate)
sd.wait()

print("Done")