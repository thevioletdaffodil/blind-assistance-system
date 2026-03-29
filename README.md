# Blind Assistance System

A real-time object detection system for visually impaired users, built with YOLOv8 and Python.

## Features
- Real-time object detection via phone camera (IP Webcam app)
- Estimated distance to each object using pinhole camera model
- Text-to-speech announcements with urgency levels
- Obstacle-free path detection (Left / Centre / Right zones)
- Approaching object detection using linear regression
- Indoor / Navigation mode toggle
- Live web dashboard — open on any phone browser on the same Wi-Fi
- Session heatmap showing where objects were most frequently detected

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Install the IP Webcam app on your Android phone and start the server

3. Update the URL in blind_assistance.py to match your phone's IP

4. Run:
   python blind_assistance.py

5. Open http://<YOUR-LAPTOP-IP>:5000 on any phone browser

## Model
Uses YOLOv8s (auto-downloaded on first run, ~22 MB).
```
