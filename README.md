# Drowsiness-Detection-System
This project detects sleeping persons in real-time images or videos
# A Python application that detects sleeping persons in images or video using eye and age detection. Works for multiple people. Shows pop-up alerts and red bounding boxes for sleepers.

## Features
- Detect multiple people
- Predict age using DeepFace
- Detect eye-closed status using Haar cascade
- Draw red (sleeping) / green (awake) rectangles
- Show pop-up alert with count and ages
- GUI to upload image or video
- Play alert sound for sleepers
# How It Works
- 1. Face Detection
Uses OpenCV's Haar Cascade classifier
Detects multiple faces in each frame
Minimum face size filtering for accuracy
- 2. Eye Analysis
Detects eyes within each face region
Calculates custom Eye Aspect Ratio (EAR)
Uses contour analysis for closed eye detection

- 3. Drowsiness Detection
Tracks consecutive frames with closed eyes
Triggers alert after 3 consecutive closed-eye frames
Visual indicators with colored bounding boxes

- 4. Age Estimation
Analyzes facial texture and features
Uses edge density and smoothness metrics
Provides age range estimates
