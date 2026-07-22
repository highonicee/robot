"""
core/intent_router.py
=====================
FIX 4 (vision classification) + FIX 2 (no LLM for intent):

- Extended _VISION_TRIGGERS to catch 'do you see X' and 'see a X' patterns
- Added _OBJECT_QUERY_TRIGGERS for 'is there a X' / 'do you see a X' patterns
- Falls back to local keyword-based intentmodule (not LLM)
- Robust import: works whether run from ROBOT/ or any subdirectory
"""

import sys
import os

# ── ROBOT/ root (parent of core/)
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VOICE_DIR  = os.path.join(_ROOT, "voice")
_ROBOT_FACE = os.path.join(_ROOT, "robot_face")
_VISION_DIR = os.path.join(_ROBOT_FACE, "vision")

for _p in [_ROOT, _VOICE_DIR, _ROBOT_FACE, _VISION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import intentmodule directly by file path so it always works
import importlib.util as _ilu

_intent_spec = _ilu.spec_from_file_location(
    "intentmodule",
    os.path.join(_VOICE_DIR, "intentmodule.py")
)
_intent_mod = _ilu.module_from_spec(_intent_spec)
_intent_spec.loader.exec_module(_intent_mod)
_original_classify = _intent_mod.classify_query

# ─────────────────────────────────────────────
# VISION KEYWORD RULES (keyword-based, instant)
# ─────────────────────────────────────────────

_VISION_TRIGGERS = [
    # "do you see" / "can you see" patterns — catches "do you see a bottle"
    "do you see", "can you see", "are you seeing", "you seeing",

    # Location
    "where is", "where are", "where do you see", "find the", "locate",
    "position of", "coordinates of",

    # Counting
    "how many", "count the", "number of objects", "how many people",
    "how many humans", "how many persons",

    # Existence check — "is there a bottle", "is there any"
    "is there a", "is there an", "is there any", "are there any",

    # Color
    "what color", "what colour", "color of", "colour of",

    # Identity
    "who do you see", "who is there", "who can you see",
    "who is in front", "who are you seeing",

    # Emotion
    "what emotion", "how does", "feeling", "mood of",

    # Action
    "what is he doing", "what is she doing",
    "what are they doing", "what is the person doing",

    # Scene / describe
    "what do you see", "what can you see", "describe what",
    "look around", "what is around", "describe the scene",
    "what is in front", "what objects",

    # Camera references
    "in camera", "on camera", "through camera", "in the camera",
    "in front of you", "in your view", "in your camera",

    # Direct vision-capability queries
    "see a", "see any", "spot a", "spot any", "detect a", "detect any",
    "notice a", "notice any", "visible", "in view",
]


def classify_query(query: str) -> str:
    """
    Returns one of: "vision", "news", "factual", "reasoning"

    Vision check runs first (keyword-based, fast).
    Falls through to keyword-based classifier if not vision.
    """
    q = query.lower().strip()

    for trigger in _VISION_TRIGGERS:
        if trigger in q:
            return "vision"

    return _original_classify(query)