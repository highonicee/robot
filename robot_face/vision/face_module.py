import numpy as np
import pickle
from insightface.app import FaceAnalysis

# Load database
with open("data/face_database.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_names = data["names"]

# Load once (IMPORTANT)
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1, det_size=(320,320))


def recognize_face(crop):

    if crop is None or crop.size == 0:
        return "Unknown"

    faces = app.get(crop)

    if len(faces) > 0:
        embedding = faces[0].embedding

        scores = np.dot(known_embeddings, embedding) / (
            np.linalg.norm(known_embeddings, axis=1) *
            np.linalg.norm(embedding)
        )

        best_index = np.argmax(scores)
        best_score = scores[best_index]

        if best_score > 0.45:
            return known_names[best_index]

    return "Unknown"