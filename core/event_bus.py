
import threading
from collections import defaultdict
from typing import Callable


class EventBus:
   

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler: Callable):
       
        with self._lock:
            self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable):
        with self._lock:
            self._subscribers[event] = [
                h for h in self._subscribers[event] if h != handler
            ]

    def publish(self, event: str, **kwargs):
       
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



class Events:
    FACE_RECOGNIZED   = "face_recognized"    
    OBJECT_DETECTED   = "object_detected"   
    PERSON_DETECTED   = "person_detected"    
    WAKE_WORD_HEARD   = "wake_word_heard"   
    QUERY_RECEIVED    = "query_received"    
    ROBOT_SPEAKING    = "robot_speaking"     
    VISION_READY      = "vision_ready"       
    SPEECH_READY      = "speech_ready"       
    SHUTDOWN          = "shutdown"           



bus = EventBus()