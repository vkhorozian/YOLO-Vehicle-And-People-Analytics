# AI NVR – YOLO Vehicle & People Analytics

## Overview

AI NVR is a lightweight Python-based Network Video Recorder (NVR) that combines real-time video analytics with event-based recording.

The application connects directly to an RTSP camera stream, performs object detection and tracking using YOLO and ByteTrack, counts vehicles and people crossing virtual barriers, detects motion from tracked objects, and automatically records video clips with metadata whenever activity occurs.

The goal is to provide a simple, self-contained analytics platform without requiring a commercial VMS.

---

## Features

### Real-Time Object Detection

* YOLOv8 object detection
* GPU acceleration using CUDA
* Configurable confidence thresholds
* Vehicle and person classification

### Object Tracking

* ByteTrack integration
* Persistent object IDs
* Motion analysis based on tracked movement
* Reduced false triggers from stationary objects

### Vehicle Counting

* Three-zone virtual barrier system
* Left-to-right counting
* Right-to-left counting
* Running totals displayed on screen

### People Counting

* Center-line crossing detection
* Direction-independent counting
* Persistent tracking IDs prevent duplicate counts

### Event Recording

* Motion-triggered recording
* Pre-event buffering
* Post-event recording extension
* MP4 video export
* JSON metadata export

### Event Metadata

Each recorded event includes:

* Timestamp
* Event type
* Vehicle counts
* Person counts
* Recording duration
* Camera source

### Operator Display

* Live video feed
* Detection overlays
* Object IDs
* Confidence scores
* Vehicle counters
* People counters
* Timestamp overlay

### Reliability Features

* Automatic RTSP reconnect
* Asynchronous video writing
* Frame buffering
* GPU availability reporting
* Stream FPS detection

---

## System Architecture

Camera (RTSP Stream)
↓
OpenCV Video Capture
↓
YOLOv8 Detection
↓
ByteTrack Tracking
↓
Motion Analysis
↓
Counting Logic
↓
Event Recording
↓
MP4 + JSON Output

---

## Requirements

### Hardware

Recommended:

* NVIDIA RTX GPU
* CUDA-compatible drivers
* 16GB+ RAM

Minimum:

* NVIDIA GTX 1050 Ti
* 8GB RAM

### Software

* Python 3.10+
* CUDA Toolkit
* OpenCV
* Ultralytics YOLO
* PyTorch

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vkhorozian/ai-nvr.git
cd ai-nvr
```

Install dependencies:

```bash
pip install ultralytics opencv-python torch torchvision
```

---

## Configuration

Edit the configuration section:

```python
RTSP_URL = "rtsp://camera-address/stream"
MODEL_PATH = "yolov8m.pt"
YOLO_CONFIDENCE = 0.55
```

Vehicle classes:

```python
VEHICLE_CLASSES = [2, 3, 5, 7]
```

Person class:

```python
PERSON_CLASS = [0]
```

---

## Running

```bash
python ai_nvr.py
```

Startup output displays:

* PyTorch version
* CUDA availability
* GPU model
* Estimated stream FPS
* YOLO processing FPS

---

## Output Structure

```text
Events_Cars/
└── event_YYYY-MM-DD_HH-MM-SS_motion
    ├── clip.mp4
    └── meta.json

Events_People/
└── event_YYYY-MM-DD_HH-MM-SS_motion
    ├── clip.mp4
    └── meta.json
```

---

## Known Limitations

This application relies heavily on ByteTrack object IDs.

If a tracked object loses its ID and receives a new one:

* Counts may be affected
* Motion detection may reset
* Event logic may restart

When troubleshooting:

1. Verify detection confidence
2. Verify tracker stability
3. Verify object persistence
4. Verify virtual barrier placement

---

## Future Enhancements

Potential future development:

* Web dashboard
* Database integration
* Multiple camera support
* License plate recognition
* Vehicle classification analytics
* Heat maps
* REST API
* MQTT integration
* Genetec integration
* Email notifications
* Telegram alerts

---

## Disclaimer

This project is intended for educational, research, and surveillance analytics purposes. Users are responsible for complying with all local laws and regulations regarding video surveillance, privacy, and data retention.

---

## License

This project is licensed under the MIT License.
