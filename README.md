#Driver Drowsiness Detection System

A real-time Driver Drowsiness Detection System built using **Python, OpenCV, and MediaPipe**.  
The system detects eye closure using Eye Aspect Ratio (EAR) and triggers a continuous alarm when drowsiness is detected.

---

#Features

- Real-time face landmark detection using MediaPipe
- Eye Aspect Ratio (EAR) calculation
- Detects prolonged eye closure
- Continuous alarm sound until eyes reopen
- Live webcam monitoring
- Lightweight and runs on CPU

---

#Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Winsound (Windows alarm system)

---

#How It Works

1. Webcam captures live video.
2. MediaPipe FaceMesh detects facial landmarks.
3. Eye landmarks are extracted.
4. Eye Aspect Ratio (EAR) is calculated.
5. If EAR falls below threshold for a fixed number of frames:
   - Drowsiness alert is displayed
   - Continuous alarm is triggered
6. Alarm stops once eyes reopen.

---
