"""
core/vision_query.py
====================
Translates natural language vision queries into answers
by reading from the shared RobotMemory.

Called by the speech intent pipeline when a "vision" intent is detected.
No hardcoded responses — all answers are built from live memory data.
"""

from core.memory import memory


def answer_vision_query(query: str) -> str:
    """
    Entry point. Takes a raw query string, returns a spoken answer string.
    Returns None if the query is not a vision query (caller should fallback).
    """
    q = query.lower().strip()

    # ── Dispatch to specific handlers ──────────────────────────────────────

    if _is_location_query(q):
        return _answer_location(q)

    if _is_count_query(q):
        return _answer_count(q)

    if _is_color_query(q):
        return _answer_color(q)

    if _is_person_query(q):
        return _answer_person(q)

    if _is_emotion_query(q):
        return _answer_emotion(q)

    if _is_action_query(q):
        return _answer_action(q)

    if _is_scene_query(q):
        return _answer_scene(q)

    # Not a vision query
    return None


# ─────────────────────────────────────────────
# INTENT DETECTORS
# ─────────────────────────────────────────────

def _is_location_query(q: str) -> bool:
    location_words = ["where", "location", "position", "coordinate", "find", "spot"]
    return any(w in q for w in location_words)

def _is_count_query(q: str) -> bool:
    count_words = ["how many", "count", "number of", "total"]
    return any(w in q for w in count_words)

def _is_color_query(q: str) -> bool:
    return "color" in q or "colour" in q

def _is_person_query(q: str) -> bool:
    return "who" in q or "person" in q or "people" in q or "human" in q

def _is_emotion_query(q: str) -> bool:
    return "emotion" in q or "feeling" in q or "mood" in q or "happy" in q or "sad" in q

def _is_action_query(q: str) -> bool:
    return "doing" in q or "action" in q or "waving" in q or "standing" in q or "sitting" in q

def _is_scene_query(q: str) -> bool:
    scene_words = ["see", "around", "scene", "environment", "detect", "visible", "front"]
    return any(w in q for w in scene_words)


# ─────────────────────────────────────────────
# ANSWER BUILDERS
# ─────────────────────────────────────────────

def _answer_location(q: str) -> str:
    """Where is the [object]? / Where do you see the [object]?"""

    # Try to extract object label from query
    label = _extract_object_label(q)

    if label:
        obj = memory.get_object_by_label(label)
        if obj:
            cx, cy = obj["center_x"], obj["center_y"]
            position_desc = _coords_to_description(cx, cy)
            return (
                f"I can see a {label} {position_desc}, "
                f"at approximately {cx} by {cy} pixels on screen."
            )
        else:
            return f"I don't currently see any {label} in front of me."

    # Check for person location
    if "person" in q or "people" in q or "human" in q or "who" in q:
        persons = memory.get_all_persons()
        if not persons:
            return "I don't see any person right now."
        parts = []
        for p in persons:
            name = p["name"] if p["name"] not in ("Unknown", "Detecting...") else "someone"
            cx, cy = p["center_x"], p["center_y"]
            parts.append(f"{name} is {_coords_to_description(cx, cy)}")
        return ". ".join(parts) + "."

    return "Could you tell me what object you're looking for?"


def _answer_count(q: str) -> str:
    """How many [objects/people] do you see?"""

    if "person" in q or "people" in q or "human" in q:
        count = memory.get_person_count()
        if count == 0:
            return "I don't see any people right now."
        elif count == 1:
            return "I can see one person in front of me."
        else:
            return f"I can see {count} people in front of me."

    label = _extract_object_label(q)
    if label:
        count = memory.get_object_count(label)
        if count == 0:
            return f"I don't see any {label} right now."
        elif count == 1:
            return f"I can see one {label}."
        else:
            return f"I can see {count} {label}s."

    # General total count
    objects = memory.get_all_objects()
    persons = memory.get_all_persons()
    total = len(objects) + len(persons)
    if total == 0:
        return "I don't see anything right now."

    obj_labels = _summarize_labels([o["label"] for o in objects])
    person_count = len(persons)

    parts = []
    if person_count:
        parts.append(f"{person_count} person{'s' if person_count > 1 else ''}")
    if obj_labels:
        parts.append(obj_labels)

    return "I can currently see: " + ", ".join(parts) + "."


def _answer_color(q: str) -> str:
    """What is the color of the [object]?"""
    label = _extract_object_label(q)
    if not label:
        return "Which object's color would you like to know?"

    obj = memory.get_object_by_label(label)
    if not obj:
        return f"I don't currently see any {label}."

    color = obj.get("color", "")
    if not color:
        return f"I can see a {label}, but I haven't detected its color yet."

    return f"The {label} appears to be {color}."


def _answer_person(q: str) -> str:
    """Who do you see?"""
    persons = memory.get_all_persons()
    if not persons:
        return "I don't see any people right now."

    names = []
    for p in persons:
        name = p["name"]
        if name in ("Unknown", "Detecting..."):
            names.append("an unknown person")
        else:
            names.append(name)

    if len(names) == 1:
        return f"I can see {names[0]} in front of me."
    else:
        return f"I can see {', '.join(names[:-1])} and {names[-1]}."


def _answer_emotion(q: str) -> str:
    """What is [person]'s emotion / how is [person] feeling?"""
    persons = memory.get_all_persons()
    if not persons:
        return "I don't see any person to read emotions from."

    # Check if a specific name is mentioned
    name_match = _extract_person_name(q, persons)
    if name_match:
        p = memory.get_person_by_name(name_match)
        if p:
            emotion = p.get("emotion", "neutral")
            return f"{name_match} appears to be feeling {emotion}."
        return f"I don't see {name_match} right now."

    # Describe everyone
    parts = []
    for p in persons:
        name = p["name"] if p["name"] not in ("Unknown", "Detecting...") else "The person"
        emotion = p.get("emotion", "neutral")
        parts.append(f"{name} looks {emotion}")
    return ". ".join(parts) + "."


def _answer_action(q: str) -> str:
    """What is [person] doing?"""
    persons = memory.get_all_persons()
    if not persons:
        return "I don't see any person right now."

    name_match = _extract_person_name(q, persons)
    if name_match:
        p = memory.get_person_by_name(name_match)
        if p:
            action = p.get("action", "standing")
            return f"{name_match} is {action}."
        return f"I don't see {name_match} right now."

    parts = []
    for p in persons:
        name = p["name"] if p["name"] not in ("Unknown", "Detecting...") else "Someone"
        action = p.get("action", "standing")
        parts.append(f"{name} is {action}")
    return ". ".join(parts) + "."


def _answer_scene(q: str) -> str:
    """What do you see? / Describe your environment."""
    objects = memory.get_all_objects()
    persons = memory.get_all_persons()

    if not objects and not persons:
        return "I don't see anything in front of me right now."

    parts = []

    if persons:
        person_parts = []
        for p in persons:
            name = p["name"] if p["name"] not in ("Unknown", "Detecting...") else "an unknown person"
            emotion = p.get("emotion", "neutral")
            action = p.get("action", "standing")
            person_parts.append(f"{name} who is {action} and looks {emotion}")
        parts.append("I can see " + ", ".join(person_parts))

    if objects:
        obj_labels = _summarize_labels([o["label"] for o in objects])
        parts.append(f"I also see {obj_labels}")

    return ". ".join(parts) + "."


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

# Common YOLO object labels for extraction
_YOLO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "stop sign", "chair", "couch",
    "potted plant", "bed", "tv", "laptop", "mouse", "keyboard", "phone",
    "book", "clock", "bottle", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "orange", "pizza", "donut", "cake", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "dog", "cat", "bird",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "cell phone", "remote", "scissors", "toothbrush", "vase", "teddy bear"
]

def _extract_object_label(q: str) -> str:
    """Scan query for known YOLO object labels."""
    q = q.lower()
    for label in _YOLO_LABELS:
        if label in q:
            return label
    return ""

def _extract_person_name(q: str, persons: list) -> str:
    """Check if any known person name appears in the query."""
    q_lower = q.lower()
    for p in persons:
        name = p.get("name", "")
        if name and name.lower() not in ("unknown", "detecting..."):
            if name.lower() in q_lower:
                return name
    return ""

def _coords_to_description(cx: int, cy: int, frame_w: int = 640, frame_h: int = 480) -> str:
    """Convert pixel coords to a human-readable description."""
    if cx < frame_w * 0.33:
        h = "on the left"
    elif cx > frame_w * 0.66:
        h = "on the right"
    else:
        h = "in the center"

    if cy < frame_h * 0.33:
        v = "top"
    elif cy > frame_h * 0.66:
        v = "bottom"
    else:
        v = "middle"

    return f"at the {v} {h}"

def _summarize_labels(labels: list) -> str:
    """Convert list of labels into 'a cup, 2 chairs, and a bottle'."""
    from collections import Counter
    counts = Counter(labels)
    parts = []
    for label, count in counts.items():
        if count == 1:
            parts.append(f"a {label}")
        else:
            parts.append(f"{count} {label}s")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]