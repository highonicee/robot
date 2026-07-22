import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# Load face database
with open("face_database.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_names = data["names"]

# Load InsightFace model (FAST version)
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1, det_size=(320,320))

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not detected")
    exit()

print(" Fast Face Recognition Running (Press Q to quit)")

frame_count = 0
faces = []

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only detect every 3 frames
    if frame_count % 3 == 0:
        faces = app.get(frame)

    for face in faces:

        embedding = face.embedding

        # Fast similarity calculation
        scores = np.dot(known_embeddings, embedding) / (
            np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
        )

        best_index = np.argmax(scores)
        best_score = scores[best_index]

        if best_score > 0.45:
            name = known_names[best_index]
        else:
            name = "Unknown"

        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            frame,
            f"{name} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Fast Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()