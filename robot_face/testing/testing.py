import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolov8n.pt")

# Load database
with open("face_database.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_names = data["names"]

# Load InsightFace
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1, det_size=(320,320))

cap = cv2.VideoCapture(0)

print("Object Detection + Face Recognition + Tracking Running")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 ONLY CHANGE: use tracking instead of normal detect
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    for r in results:

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        # 🔥 NEW: get tracking IDs
        track_ids = (
            r.boxes.id.cpu().numpy()
            if r.boxes.id is not None else [None]*len(boxes)
        )

        for box, cls, conf, track_id in zip(boxes, classes, confs, track_ids):

            x1, y1, x2, y2 = map(int, box)

            label = model.names[int(cls)]
            name = label

            # Limit box inside frame
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # ---------------- FACE RECOGNITION (UNCHANGED) ---------------- #
            if label == "person":

                crop = frame[y1:y2, x1:x2]

                if crop.size != 0:

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
                            name = known_names[best_index]
                        else:
                            name = "Unknown Person"

            # ---------------- DRAW ---------------- #

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # 🔥 NEW: show tracking ID
            cv2.putText(
                frame,
                f"ID:{track_id} {name} {conf:.2f}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    cv2.imshow("YOLO Detection + Face Recognition + Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()