print("Emotion module loaded ✅")

import cv2

def detect_emotion(crop):

    if crop is None or crop.size == 0:
        return ""

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Simple brightness-based heuristic
        brightness = gray.mean()

        if brightness > 140:
            return "happy"
        elif brightness < 90:
            return "sad"
        else:
            return ""

    except:
        return ""