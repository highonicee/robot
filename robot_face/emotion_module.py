try:
    from fer import FER
    emotion_model = FER(mtcnn=False)
    EMOTION_OK = True
except:
    EMOTION_OK = False


def detect_emotion(crop):

    if not EMOTION_OK:
        return ""

    if crop is None or crop.size == 0:
        return ""

    try:
        emotions = emotion_model.detect_emotions(crop)

        if len(emotions) > 0:
            return max(
                emotions[0]["emotions"],
                key=emotions[0]["emotions"].get
            )
    except:
        return ""

    return ""