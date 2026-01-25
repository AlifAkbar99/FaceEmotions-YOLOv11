import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO classification model
model = YOLO(r'D:\machine learning\emotions\models\last.pt')

# Webcam
video = cv2.VideoCapture(0)

# Face detector
faceDetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]   # YOLO butuh RGB/BGR, BUKAN grayscale

        # YOLO classification
        result = model(face_img, verbose=False)[0]

        class_id = int(result.probs.top1)
        confidence = result.probs.top1conf.item()
        label = model.names[class_id]

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 0, 255), -1)

        # Text
        cv2.putText(
            frame,
            f'{label} {confidence:.2f}',
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Emotion Detection (YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
