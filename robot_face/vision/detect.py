"""
vision/detect.py
================
verbose=False silences all YOLO terminal output.
"""
from ultralytics import YOLO
import os

_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "yolov8n.pt"
)
model = YOLO(_MODEL)

def detect_objects(frame):
    return model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False      # ← suppresses all the "0: 480x640 1 person" lines
    )