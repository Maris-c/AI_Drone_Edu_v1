# AI Drone Gesture Recognition - Training Version 1

## 📌 Overview
Version 1 focuses on training a YOLOv8 model to recognize hand gestures for controlling a drone. The model detects gestures from a laptop webcam and outputs the corresponding gesture class for later integration with the drone control system. :contentReference[oaicite:0]{index=0}

## 🎯 Objectives
- Collect and label gesture images.
- Train a YOLOv8 gesture recognition model.
- Evaluate model performance.
- Export the trained model (`best.pt`) for real-time testing. :contentReference[oaicite:1]{index=1}

## 🖐️ Gesture Classes
- Open Hand → Takeoff
- Fist → Land
- One Finger → Forward
- Two Fingers → Backward
- Three Fingers → Right
- Four Fingers → Left
- Thumb Up → Up
- Thumb Down → Down
- Shaka → Hover / Emergency Stop :contentReference[oaicite:2]{index=2}

## 🛠️ Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Roboflow (Dataset Management)

## 📂 Project Structure
```
dataset/
models/
runs/
train.py
predict.py
requirements.txt
README.md
```

## 🚀 Training
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## 📊 Output
- `best.pt` – Best trained model
- `last.pt` – Last training checkpoint
- Training metrics (Precision, Recall, mAP)

## 🔜 Next Version
- Improve dataset quality.
- Increase recognition accuracy.
- Integrate with MAVLink for real-time drone control. :contentReference[oaicite:3]{index=3}
