import cv2

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # change number if needed
  # change number if needed

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Frame not received")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()