"""
core/event_bus.py
=================
Lightweight publish/subscribe event bus.

Vision thread  → publishes events (face_recognized, object_detected, etc.)
Speech thread  → subscribes to events (to trigger greetings, alerts, etc.)

No external dependencies — pure Python threading.
"""

import threading
from collections import defaultdict
from typing import Callable


class EventBus:
    """
    Thread-safe pub/sub event bus.

    Usage:
        bus.subscribe("face_recognized", my_handler)
        bus.publish("face_recognized", name="Arshiya", emotion="happy")
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler: Callable):
        """Register a handler for an event."""
        with self._lock:
            self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable):
        with self._lock:
            self._subscribers[event] = [
                h for h in self._subscribers[event] if h != handler
            ]

    def publish(self, event: str, **kwargs):
        """
        Fire an event. Handlers run in separate daemon threads
        so they never block the publisher (vision loop stays fast).
        """
        with self._lock:
            handlers = list(self._subscribers.get(event, []))

        for handler in handlers:
            threading.Thread(
                target=self._safe_call,
                args=(handler, event, kwargs),
                daemon=True
            ).start()

    def _safe_call(self, handler: Callable, event: str, kwargs: dict):
        try:
            handler(**kwargs)
        except Exception as e:
            print(f"[EventBus] Handler error on event '{event}': {e}")


# ─────────────────────────────────────────────
# DEFINED EVENTS (treat as constants)
# ─────────────────────────────────────────────
class Events:
    FACE_RECOGNIZED   = "face_recognized"    # kwargs: name, emotion, action, bbox
    OBJECT_DETECTED   = "object_detected"    # kwargs: label, bbox, track_id
    PERSON_DETECTED   = "person_detected"    # kwargs: track_id, name, emotion
    WAKE_WORD_HEARD   = "wake_word_heard"    # kwargs: (none)
    QUERY_RECEIVED    = "query_received"     # kwargs: query (str)
    ROBOT_SPEAKING    = "robot_speaking"     # kwargs: text (str)
    VISION_READY      = "vision_ready"       # kwargs: (none)
    SPEECH_READY      = "speech_ready"       # kwargs: (none)
    SHUTDOWN          = "shutdown"           # kwargs: (none)


# ─────────────────────────────────────────────
# GLOBAL SINGLETON
# ─────────────────────────────────────────────
bus = EventBus()