from ultralytics import YOLO

# Load pose model
pose_model = YOLO("models/yolov8n-pose.pt")

def detect_pose(frame):
    # FIX 1: verbose=False suppresses the "0: 480x640 1 person, 45.6ms" spam
    results = pose_model(frame, verbose=False)
    return results

def get_action_from_keypoints(keypoints):
    """
    Simple action logic using keypoints
    """
    if keypoints is None or len(keypoints) == 0:
        return ""

    try:
        kp = keypoints[0]  # first person

        # Keypoints: [x, y, confidence]
        nose        = kp[0]
        left_wrist  = kp[9]
        right_wrist = kp[10]
        left_hip    = kp[11]
        right_hip   = kp[12]

        # Waving detection
        if left_wrist[1] < nose[1] or right_wrist[1] < nose[1]:
            return "waving"

        # Sitting detection
        if left_hip[1] < nose[1] and right_hip[1] < nose[1]:
            return "sitting"

        # Moving / standing
        return "standing"

    except:
        return ""