import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# -------------------------------
# Eye Aspect Ratio Function
# -------------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


print("Initializing MediaPipe...")

# -------------------------------
# Initialize MediaPipe FaceMesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Starting camera...")

# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()
else:
    print("✅ Camera detected")

# -------------------------------
# Eye Landmark Indexes (MediaPipe)
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
FRAME_COUNTER = 0
ALARM_THRESHOLD = 20

print("System Running... Press Q to Quit")

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye = []
            right_eye = []

            # Left Eye Points
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))

            # Right Eye Points
            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))

            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye landmarks
            for point in left_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            for point in right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Drowsiness Detection
            if ear < EAR_THRESHOLD:
                FRAME_COUNTER += 1

                if FRAME_COUNTER >= ALARM_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 0, 255),
                                3)
            else:
                FRAME_COUNTER = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()