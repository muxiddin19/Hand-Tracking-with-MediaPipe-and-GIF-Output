# Hand-Tracking-with-MediaPipe-and-GIF-Output
This repository contains a Python script that uses MediaPipe to perform real-time hand tracking via a webcam and saves the output as a GIF. The script leverages OpenCV for video capture and ImageIO for GIF creation, making it a lightweight solution for visualizing hand movements.

# Features
- Detects and tracks hands in real-time using MediaPipe’s Hands solution.

- Draws hand landmarks and connections on webcam frames.

- Saves the annotated video feed as a GIF file (hand_tracking_output.gif).

- Automatically detects a working camera index.

- Configurable frame limit and GIF frame rate.

# Prerequisites
- Python 3.9+ (tested with 3.9)

- A webcam connected to your system

- Linux (e.g., Ubuntu) recommended; adaptable to other OS with minor tweaks

# Installation
## 1. Clone the Repository
```bash
git clone https://github.com/yourusername/hand-tracking-gif.git
cd hand-tracking-gif
```
## 2. Set Up a Conda Environment
```bash
# Create a new Conda environment
conda create -n mp python=3.9
conda activate mp
```
## 3. Install Dependencies
Install the required Python packages using the provided requirements.txt:
```bash

pip install -r requirements.txt
```
requirements.txt
```bash
mediapipe==0.10.18
opencv-contrib-python>=4.5.5
imageio>=2.9.0
numpy>=1.21.0
absl-py>=1.0.0
attrs>=21.0.0
flatbuffers>=2.0
jax>=0.4.0
jaxlib>=0.4.0
matplotlib>=3.5.0
protobuf>=3.20.0
sentencepiece>=0.1.96
sounddevice>=0.4.4
```
## 4. Verify Camera Access (Linux)
Ensure your webcam is detected and accessible:
```bash

ls /dev/video*
```
- Expected output: /dev/video0, /dev/video1, etc.

- Fix permissions if needed:
```bash
sudo chmod 666 /dev/video0
sudo chmod 666 /dev/video1
```
# Usage
### 1. Run the Script:
```bash

python hand_tracker.py
```
- The script will:
-- Search for a working camera (indices 0–2).

-- Display a window with real-time hand tracking.

-- Save a GIF after capturing 100 frames or when you press 'q'.

### 2. Output:
- A file named hand_tracking_output.gif will be saved in the current directory.

- GIF duration is set to 0.1 seconds per frame (10 FPS).

### 3. Controls:
- Press q to stop tracking and save the GIF manually.

- The script auto-stops after 100 frames if q isn’t pressed.

# Script Overview (hand_tracker.py)
```python

import cv2
import mediapipe as mp
import imageio
import numpy as np

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Camera detection function
def find_camera(max_index=3):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            print(f"Using camera at index {i}")
            return cap, i
        cap.release()
    print("No working camera found. Exiting.")
    exit(1)

# Main logic
cap, camera_index = find_camera()
frames = []
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read frame.")
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow('MediaPipe Hands', image)
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(5) & 0xFF == ord('q') or len(frames) >= 100:
            break

# Save GIF
if frames:
    imageio.mimsave('hand_tracking_output.gif', frames, duration=0.1)
    print("GIF saved as hand_tracking_output.gif")
cap.release()
cv2.destroyAllWindows()
```
# Customization
- Frame Limit: Change len(frames) >= 100 to adjust the GIF length.

- GIF Speed: Modify duration=0.1 in imageio.mimsave() (e.g., 0.05 for 20 FPS).

- Camera Index: Edit max_index in find_camera() if you have more devices.

# Troubleshooting
- "No frames captured to save as GIF":
-- Ensure your webcam is connected and not in use by another app (lsof /dev/video0).

-- Test camera indices:
```python
import cv2
for i in range(3):
    cap = cv2.VideoCapture(i)
    print(f"Index {i}: {'Open' if cap.isOpened() else 'Failed'}")
    cap.release()
```
- Permissions Error:
-- Add your user to the video group:
```bash

sudo usermod -aG video $USER
```
Then reboot.

- ModuleNotFoundError:
-- Verify all dependencies are installed (pip list | grep <package>).

# Contributing
Feel free to fork this repo, submit issues, or send pull requests to improve functionality (e.g., adding static image support, multi-hand tracking options).
# License
This project is licensed under the MIT License. See LICENSE for details.
# Acknowledgments
Built with MediaPipe by Google.
``bash
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
```

Thanks to the open-source community for tools like OpenCV and ImageIO.


