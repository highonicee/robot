

import sys
import os

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VOICE_DIR  = os.path.join(_ROOT, "voice")
_ROBOT_FACE = os.path.join(_ROOT, "robot_face")
_VISION_DIR = os.path.join(_ROBOT_FACE, "vision")

for _p in [_ROOT, _VOICE_DIR, _ROBOT_FACE, _VISION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


import importlib.util as _ilu

_intent_spec = _ilu.spec_from_file_location(
    "intentmodule",
    os.path.join(_VOICE_DIR, "intentmodule.py")
)
_intent_mod = _ilu.module_from_spec(_intent_spec)
_intent_spec.loader.exec_module(_intent_mod)
_original_classify = _intent_mod.classify_query



_VISION_TRIGGERS = [
  
    "do you see", "can you see", "are you seeing", "you seeing",

    
    "where is", "where are", "where do you see", "find the", "locate",
    "position of", "coordinates of",

    
    "how many", "count the", "number of objects", "how many people",
    "how many humans", "how many persons",

    "is there a", "is there an", "is there any", "are there any",

  
    "what color", "what colour", "color of", "colour of",

  
    "who do you see", "who is there", "who can you see",
    "who is in front", "who are you seeing",

   
    "what emotion", "how does", "feeling", "mood of",

   
    "what is he doing", "what is she doing",
    "what are they doing", "what is the person doing",

  
    "what do you see", "what can you see", "describe what",
    "look around", "what is around", "describe the scene",
    "what is in front", "what objects",

  
    "in camera", "on camera", "through camera", "in the camera",
    "in front of you", "in your view", "in your camera",

    
    "see a", "see any", "spot a", "spot any", "detect a", "detect any",
    "notice a", "notice any", "visible", "in view",
]


def classify_query(query: str) -> str:
    
    q = query.lower().strip()

    for trigger in _VISION_TRIGGERS:
        if trigger in q:
            return "vision"

    return _original_classify(query)