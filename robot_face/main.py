import cv2
from vision.detect import detect_objects
from vision.face_module import recognize_face
from vision.emotion_module import detect_emotion
from insightface.app import FaceAnalysis
from vision.pose import detect_pose, get_action_from_keypoints
from voice.tts import speak

# Face detector
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=-1, det_size=(320, 320))

cap = cv2.VideoCapture(0)

frame_count = 0

# Cache
face_cache = {}
emotion_cache = {}    
action_cache = {}

pose_results = None

print("🚀 Vision System Running...")

# ✅ SPEAK AFTER EVERYTHING IS READY
speak("Hello, my name is Mini Dora. How can I help you today?")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (640, 480))

    # 🔥 Run object detection
    results = detect_objects(frame)

    # 🔥 Run pose every 10 frames (optimized)
    if frame_count % 10 == 0:
        pose_results = detect_pose(frame)

    for r in results:

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [-1]*len(boxes)
        classes = r.boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)
            label = r.names[int(cls)]

            display = label

        
            if label == "person":

                name = "Detecting..."
                emotion = "neutral"
                action = ""

                person_crop = frame[y1:y2, x1:x2]

                # 🔹 FACE DETECTION (every 5 frames)
                if frame_count % 5 == 0:
                    faces = face_app.get(person_crop)
                else:
                    faces = []

                if len(faces) > 0:

                    face = faces[0]
                    fx1, fy1, fx2, fy2 = map(int, face.bbox)
                    face_crop = person_crop[fy1:fy2, fx1:fx2]

                    # 🔹 FACE RECOGNITION
                    if frame_count % 10 == 0:
                        name = recognize_face(face_crop)
                        face_cache[track_id] = name
                    else:
                        name = face_cache.get(track_id, "Detecting...")

                    # 🔹 EMOTION
                    if frame_count % 20 == 0:
                        emotion = detect_emotion(face_crop)
                        if emotion == "":
                            emotion = "neutral"
                        emotion_cache[track_id] = emotion
                    else:
                        emotion = emotion_cache.get(track_id, "neutral")

                    # 🔥 REAL POSE ACTION
                    if pose_results is not None:

                        for pr in pose_results:

                            if pr.boxes is None or pr.keypoints is None:
                                continue

                            pose_boxes = pr.boxes.xyxy.cpu().numpy()
                            keypoints = pr.keypoints.xy.cpu().numpy()

                            # 🔥 MATCH POSE PERSON WITH YOLO PERSON (IoU-like check)
                            for pb, kp in zip(pose_boxes, keypoints):

                                px1, py1, px2, py2 = map(int, pb)

                                # simple overlap check
                                if (abs(px1 - x1) < 50 and abs(py1 - y1) < 50):

                                    action = get_action_from_keypoints([kp])
                                    if action != "":
                                        action_cache[track_id] = action
                                    break

                    action = action_cache.get(track_id, "")

                    # Draw face box
                    cv2.rectangle(frame,
                                  (x1 + fx1, y1 + fy1),
                                  (x1 + fx2, y1 + fy2),
                                  (255, 0, 0), 2)

                display = f"{name} : {emotion} : {action}"

            # 🔥 DRAW ALL OBJECTS
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, display,
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

    cv2.imshow("Humanoid Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()