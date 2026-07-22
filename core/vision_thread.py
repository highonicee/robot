"""
core/vision_thread.py
=====================
Changes from original:
- ALL terminal print/logging from YOLO suppressed (verbose=False)
- Vision runs in its own thread — does not block speech
- Whisper CPU spike fix: vision skips heavy processing during STT
- Color detection, pose, face unchanged
"""

import cv2
import threading
import time
import numpy as np
import sys
import os

_CORE_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_CORE_DIR, ".."))
_ROBOT_FACE = os.path.join(_ROOT, "robot_face")
_VISION_DIR = os.path.join(_ROBOT_FACE, "vision")
_VOICE_DIR  = os.path.join(_ROOT, "voice")
for _p in [_ROOT, _VOICE_DIR, _ROBOT_FACE, _VISION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.memory import memory
from core.event_bus import bus, Events
from robot_face.vision.detect import detect_objects
from robot_face.vision.face_module import recognize_face
from robot_face.vision.emotion_module import detect_emotion
from robot_face.vision.pose import detect_pose, get_action_from_keypoints
from insightface.app import FaceAnalysis


# ── Global flag: speech thread sets this True while Whisper is running ────────
# Vision thread skips heavy work (pose, face, emotion) during that window
_stt_running = threading.Event()

def notify_stt_start():
    """Call this from speech thread when Whisper starts."""
    _stt_running.set()

def notify_stt_end():
    """Call this from speech thread when Whisper finishes."""
    _stt_running.clear()


class VisionThread:

    def __init__(self, camera_index: int = 0, show_window: bool = True):
        self.camera_index = camera_index
        self.show_window  = show_window

        self._thread       = None
        self._stop_event   = threading.Event()
        self._ready_event  = threading.Event()

        self._face_cache   = {}
        self._emotion_cache= {}
        self._action_cache = {}
        self._pose_results = None

        self._face_app = FaceAnalysis(name="buffalo_s")
        self._face_app.prepare(ctx_id=-1, det_size=(320, 320))

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="VisionThread", daemon=True)
        self._thread.start()
        print("[Vision] Thread started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[Vision] Thread stopped.")

    def wait_until_ready(self, timeout: float = 15.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[Vision] ERROR: Could not open camera.")
            return

        frame_count      = 0
        stale_flush_timer = time.time()

        self._ready_event.set()
        bus.publish(Events.VISION_READY)
        print("[Vision] Ready.")

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (640, 480))

            # ── YOLO detection — verbose=False suppresses all terminal output ──
            results = detect_objects(frame)

            # ── Skip heavy processing while Whisper is using CPU ──────────────
            stt_busy = _stt_running.is_set()

            if not stt_busy and frame_count % 10 == 0:
                self._pose_results = detect_pose(frame)

            if time.time() - stale_flush_timer > 3.0:
                memory.flush_stale()
                stale_flush_timer = time.time()

            total_objects = 0
            total_persons = 0

            for r in results:
                if r.boxes is None:
                    continue

                boxes   = r.boxes.xyxy.cpu().numpy()
                ids     = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [-1]*len(boxes)
                classes = r.boxes.cls.cpu().numpy()
                confs   = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else [0.0]*len(boxes)

                for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    label    = r.names[int(cls)]
                    track_id = int(track_id)

                    if label == "person":
                        total_persons += 1
                        # Skip face/emotion/pose while STT is running
                        if not stt_busy:
                            self._process_person(frame, track_id, x1, y1, x2, y2, frame_count)
                        else:
                            # Still update memory with last known values
                            cached = memory._persons.get(track_id)
                            if cached:
                                memory.update_person(
                                    track_id, cached["name"],
                                    cached["emotion"], cached["action"],
                                    (x1, y1, x2, y2)
                                )
                    else:
                        total_objects += 1
                        color = self._detect_color(frame, x1, y1, x2, y2)
                        memory.update_object(
                            track_id=track_id, label=label,
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf), color=color
                        )

                    self._draw_box(frame, label, track_id, x1, y1, x2, y2)

            memory.update_scene(total_objects, total_persons)

            if self.show_window:
                cv2.imshow("Mini Dora — Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop_event.set()
                    break

        cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        bus.publish(Events.SHUTDOWN)

    def _process_person(self, frame, track_id, x1, y1, x2, y2, frame_count):
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return

        name    = self._face_cache.get(track_id, "Detecting...")
        emotion = self._emotion_cache.get(track_id, "neutral")
        action  = self._action_cache.get(track_id, "standing")

        if frame_count % 5 == 0:
            faces = self._face_app.get(person_crop)
        else:
            faces = []

        if faces:
            face = faces[0]
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(person_crop.shape[1], fx2), min(person_crop.shape[0], fy2)
            face_crop = person_crop[fy1:fy2, fx1:fx2]

            if face_crop.size > 0:
                if frame_count % 10 == 0:
                    new_name = recognize_face(face_crop)
                    self._face_cache[track_id] = new_name
                    name = new_name
                    if name not in ("Unknown", "Detecting...") and memory.should_greet(name):
                        bus.publish(Events.FACE_RECOGNIZED,
                                    name=name, emotion=emotion,
                                    action=action, bbox=(x1,y1,x2,y2))

                if frame_count % 20 == 0:
                    detected = detect_emotion(face_crop)
                    if detected:
                        self._emotion_cache[track_id] = detected
                        emotion = detected

                cv2.rectangle(frame,
                    (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), (255,0,0), 2)

        if self._pose_results is not None:
            action = self._match_pose_action(track_id, x1, y1, action)

        memory.update_person(track_id, name, emotion, action, (x1,y1,x2,y2))

    def _match_pose_action(self, track_id, x1, y1, fallback):
        for pr in self._pose_results:
            if pr.boxes is None or pr.keypoints is None:
                continue
            for pb, kp in zip(pr.boxes.xyxy.cpu().numpy(), pr.keypoints.xy.cpu().numpy()):
                px1, py1 = int(pb[0]), int(pb[1])
                if abs(px1-x1) < 50 and abs(py1-y1) < 50:
                    detected = get_action_from_keypoints([kp])
                    if detected:
                        self._action_cache[track_id] = detected
                        return detected
        return self._action_cache.get(track_id, fallback)

    def _detect_color(self, frame, x1, y1, x2, y2):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mh, ms, mv = int(np.mean(h)), int(np.mean(s)), int(np.mean(v))
            if ms < 40:
                return "black" if mv < 60 else "white" if mv > 200 else "gray"
            if mh < 10 or mh >= 160: return "red"
            if mh < 25:  return "orange"
            if mh < 35:  return "yellow"
            if mh < 85:  return "green"
            if mh < 125: return "blue"
            if mh < 145: return "purple"
            if mh < 160: return "pink"
        except Exception:
            pass
        return ""

    def _draw_box(self, frame, label, track_id, x1, y1, x2, y2):
        if label == "person":
            p = memory._persons.get(track_id, {})
            name    = p.get("name", "?")
            emotion = p.get("emotion", "")
            action  = p.get("action", "")
            display = f"{name} | {emotion} | {action}"
        else:
            o = memory._objects.get(track_id, {})
            color = o.get("color", "")
            display = f"{label}" + (f" ({color})" if color else "")

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, display, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)