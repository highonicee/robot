"""
core/memory.py
==============
Professional shared memory store for Mini Dora robot.
Uses SQLite for persistence + in-memory cache for real-time speed.

Written to: Store vision detections (objects, persons, emotions, actions)
Read from:  Speech module to answer natural language queries
"""

import sqlite3
import threading
import time
import json
import os
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_memory.db")
STALE_THRESHOLD_SECONDS = 5  # Objects not seen for 5s are considered gone


# ─────────────────────────────────────────────
# THREAD-SAFE SINGLETON
# ─────────────────────────────────────────────
_lock = threading.Lock()


class RobotMemory:
    """
    Thread-safe memory store for the robot.

    Two layers:
        1. In-memory cache  → fast real-time reads (vision loop writes here)
        2. SQLite DB        → persistent logs (for history, greetings, etc.)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()

        # ── Live cache (vision writes, speech reads) ──
        self._objects: dict[int, dict] = {}   # track_id → object data
        self._persons: dict[int, dict] = {}   # track_id → person data
        self._scene: dict = {}                # global scene summary

        # ── Greeting tracking (persists in DB) ──
        self._greeted_today: set[str] = set()

        self._init_db()
        self._load_greeted_today()

    # ─────────────────────────────────────────
    # DB SETUP
    # ─────────────────────────────────────────
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    track_id    INTEGER,
                    bbox        TEXT,
                    center_x    INTEGER,
                    center_y    INTEGER,
                    confidence  REAL,
                    color       TEXT,
                    extra       TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    track_id    INTEGER,
                    name        TEXT,
                    emotion     TEXT,
                    action      TEXT,
                    bbox        TEXT,
                    center_x    INTEGER,
                    center_y    INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS greet_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    greeted_at  TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_greeted_today(self):
        """Load names already greeted today so we don't repeat on restart."""
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM greet_log WHERE greeted_at LIKE ?",
                (f"{today}%",)
            ).fetchall()
        self._greeted_today = {row[0] for row in rows}

    # ─────────────────────────────────────────
    # WRITE: VISION → MEMORY
    # ─────────────────────────────────────────
    def update_object(
        self,
        track_id: int,
        label: str,
        bbox: tuple,
        confidence: float = 0.0,
        color: str = "",
        extra: dict = None
    ):
        """Called by vision loop for every non-person detection."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        entry = {
            "track_id":  track_id,
            "label":     label,
            "bbox":      bbox,
            "center_x":  center_x,
            "center_y":  center_y,
            "confidence": confidence,
            "color":     color,
            "extra":     extra or {},
            "last_seen": time.time(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        with self._lock:
            self._objects[track_id] = entry

        # Async DB write (don't block vision loop)
        threading.Thread(
            target=self._persist_object, args=(entry,), daemon=True
        ).start()

    def update_person(
        self,
        track_id: int,
        name: str,
        emotion: str,
        action: str,
        bbox: tuple
    ):
        """Called by vision loop for every person detection."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        entry = {
            "track_id":  track_id,
            "name":      name,
            "emotion":   emotion,
            "action":    action,
            "bbox":      bbox,
            "center_x":  center_x,
            "center_y":  center_y,
            "last_seen": time.time(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        with self._lock:
            self._persons[track_id] = entry

        threading.Thread(
            target=self._persist_person, args=(entry,), daemon=True
        ).start()

    def update_scene(self, total_objects: int, total_persons: int):
        """Called by vision loop after processing each frame."""
        with self._lock:
            self._scene = {
                "total_objects": total_objects,
                "total_persons": total_persons,
                "timestamp":     datetime.now().isoformat(timespec="seconds"),
            }

    # ─────────────────────────────────────────
    # READ: SPEECH → MEMORY
    # ─────────────────────────────────────────
    def get_all_objects(self, stale_ok: bool = False) -> list[dict]:
        """Returns all currently visible objects (fresh only by default)."""
        now = time.time()
        with self._lock:
            if stale_ok:
                return list(self._objects.values())
            return [
                o for o in self._objects.values()
                if now - o["last_seen"] < STALE_THRESHOLD_SECONDS
            ]

    def get_object_by_label(self, label: str) -> Optional[dict]:
        """Find the most recently seen object matching a label."""
        label = label.lower()
        matches = [
            o for o in self.get_all_objects()
            if o["label"].lower() == label
        ]
        if not matches:
            return None
        return max(matches, key=lambda o: o["last_seen"])

    def get_all_persons(self, stale_ok: bool = False) -> list[dict]:
        """Returns all currently visible persons."""
        now = time.time()
        with self._lock:
            if stale_ok:
                return list(self._persons.values())
            return [
                p for p in self._persons.values()
                if now - p["last_seen"] < STALE_THRESHOLD_SECONDS
            ]

    def get_person_by_name(self, name: str) -> Optional[dict]:
        """Find a person by recognized name."""
        name = name.lower()
        for p in self.get_all_persons():
            if p["name"].lower() == name:
                return p
        return None

    def get_scene(self) -> dict:
        with self._lock:
            return dict(self._scene)

    def get_object_count(self, label: str = None) -> int:
        """Count visible objects, optionally filtered by label."""
        objects = self.get_all_objects()
        if label:
            label = label.lower()
            return sum(1 for o in objects if o["label"].lower() == label)
        return len(objects)

    def get_person_count(self) -> int:
        return len(self.get_all_persons())

    # ─────────────────────────────────────────
    # GREETING SYSTEM
    # ─────────────────────────────────────────
    def should_greet(self, name: str) -> bool:
        """
        Returns True if this person should be greeted.
        Greets each recognized person only ONCE per session.
        Unknown/Detecting persons are never greeted.
        """
        if not name or name.lower() in ("unknown", "detecting...", ""):
            return False
        return name not in self._greeted_today

    def mark_greeted(self, name: str):
        """Mark a person as greeted so we don't repeat."""
        self._greeted_today.add(name)
        timestamp = datetime.now().isoformat(timespec="seconds")
        threading.Thread(
            target=self._persist_greet, args=(name, timestamp), daemon=True
        ).start()

    # ─────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────
    def flush_stale(self):
        """Remove stale detections from live cache. Call periodically."""
        now = time.time()
        with self._lock:
            self._objects = {
                tid: o for tid, o in self._objects.items()
                if now - o["last_seen"] < STALE_THRESHOLD_SECONDS
            }
            self._persons = {
                tid: p for tid, p in self._persons.items()
                if now - p["last_seen"] < STALE_THRESHOLD_SECONDS
            }

    # ─────────────────────────────────────────
    # PRIVATE: ASYNC DB WRITES
    # ─────────────────────────────────────────
    def _persist_object(self, entry: dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO detections
                    (timestamp, label, track_id, bbox, center_x, center_y, confidence, color, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry["timestamp"],
                    entry["label"],
                    entry["track_id"],
                    json.dumps(entry["bbox"]),
                    entry["center_x"],
                    entry["center_y"],
                    entry["confidence"],
                    entry["color"],
                    json.dumps(entry["extra"]),
                ))
                conn.commit()
        except Exception as e:
            print(f"[Memory] DB write error (object): {e}")

    def _persist_person(self, entry: dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO persons
                    (timestamp, track_id, name, emotion, action, bbox, center_x, center_y)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry["timestamp"],
                    entry["track_id"],
                    entry["name"],
                    entry["emotion"],
                    entry["action"],
                    json.dumps(entry["bbox"]),
                    entry["center_x"],
                    entry["center_y"],
                ))
                conn.commit()
        except Exception as e:
            print(f"[Memory] DB write error (person): {e}")

    def _persist_greet(self, name: str, timestamp: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO greet_log (name, greeted_at) VALUES (?, ?)",
                    (name, timestamp)
                )
                conn.commit()
        except Exception as e:
            print(f"[Memory] DB write error (greet): {e}")


# ─────────────────────────────────────────────
# GLOBAL SINGLETON (import this everywhere)
# ─────────────────────────────────────────────
memory = RobotMemory()