# The Aerial Guardian

An advanced, edge-optimized Multi-Object Tracking (MOT) pipeline designed specifically for high-altitude drone footage. Built to overcome the unique computer vision hurdles of UAV platforms: extremely small target sizes, heavy camera rotation/panning (ego-motion), and edge-hardware CPU constraints.

Output videos: https://drive.google.com/drive/folders/1HvsfIYDQLJFm8PuLxsavAKMFKYiBnaUD?usp=sharing

## 🚀 Key Features & Architecture

### 1. High-Altitude Detection: Slicing Aided Hyper Inference (SAHI)
Drones capture massive resolution imagery where targets (cars, people) may only be 10-15 pixels wide. Standard downscaling completely destroys these features.
- We utilize **SAHI** with a `yolov8` backend (`yolo11n.onnx`). 
- By mathematically slicing a 720p base resolution frame into exactly 4 non-overlapping `640x640` patches, we process tiny targets at their native resolution without redundant CPU overload.

### 2. Edge Hardware Engineering: Inference Throttling
Running native neural networks frame-by-frame on CPU edge devices (like a Ryzen 3 or Raspberry Pi) yields `<1 FPS`.
- To counter this, we implemented an **Inference Interval** loop. YOLO inference only executes once every 4 frames.
- For the skipped frames, we bypass the AI engine entirely and extrapolate positions using an "Artificial Life Support" Kalman loop, pushing **>120 FPS** on intermediate frames to dramatically smooth the video output.

### 3. Drone Ego-Motion Mitigation
Drones constantly pan and rotate, which causes standard trackers to glitch, creating "ghost boxes" or heavily zigzagging trajectory tails.
- We implemented Norfair's **MotionEstimator** utilizing a **Homography Transformation**.
- This calculates affine background drift using OpenCV CPU logic, instantly anchoring all tracking predictions against the drone's rotation and zoom.

### 4. End-to-End Analytics Layer
We expanded the pipeline beyond simple bounding boxes to extract real-world telemetry:
- **Persistent Thermal Heatmaps:** A Numpy matrix automatically paints high-density traffic areas. A thermal decay rate naturally cools down older traffic zones.
- **Speed Estimation:** Because the drone's ego-motion is mathematically isolated, the pipeline accurately calculates the *true* physical pixel speed (`px/s`) of tracked vehicles.

## 📊 Performance Metrics
- **Hardware Target**: CPU-Only Edge Devices (Ryzen 3, 8GB RAM)
- **Model Size**: Ultralytics YOLOv11 Nano (~6 MB)
- **Throttled Interpolation Speed**: ~120 FPS
- **Tracking Engine**: Norfair (Kalman Filter-based)
- **Base Resolution Constraint**: 1280x720 (Exactly four 640x640 inference patches)

## 💻 Quick Start

**1. Install Dependencies**
```bash
pip install -r requirements.txt
pip install onnx openvino gdown
```

**2. Download the Dataset**
```bash
gdown 1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu
unzip VisDrone2019-MOT-val.zip
```

**3. Compile Models**
Standard PyTorch (`.pt`) is not optimized for Edge CPUs. Run the compiler script to natively convert the weights to C++ architectures:
```bash
python export_model.py
```

**4. Execute the Pipeline**
```bash
python sahi_pipeline.py
```
