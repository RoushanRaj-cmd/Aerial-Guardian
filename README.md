# Aerial Guardian: Edge-Optimized Multi-Object Tracking Pipeline

This repository contains the setup and summary report for "The Aerial Guardian" challenge. We built a lightweight, edge-optimized pipeline designed to detect and track targets (Persons and Vehicles) from high-altitude moving drone platforms, specifically addressing minute target sizes and severe camera ego-motion.

## Technical Completion Checklist

- [x] **Dataset:** VisDrone2019 MOT Validation Set.
- [x] **Target Classes:** Persons & Vehicles.
- [x] **Lightweight Base Model (< 300MB):** We utilize the state-of-the-art **YOLO11 Nano** model. The raw weights (`yolo11n.pt`) are incredibly lightweight (**~6 MB**), well within the 300 MB limit, ensuring a minimal memory footprint on edge devices while maintaining robust detection.
- [x] **Detection Optimization:** Slicing Aided Hyper Inference (SAHI) implemented for small-target sensitivity.
- [x] **Tracking Optimization:** Norfair MOT utilized alongside Optical Flow Camera Motion Compensation (CMC) to handle drone motion and robust ID assignment.
- [x] **Output Video Status:** Generates rendering with bounding boxes, unique ID labels, and yellow trajectory "tails" tracing path history.

## Setup & Execution

### Prerequisites
Requires Python 3.8+ and standard dependencies:
```bash
pip install ultralytics sahi norfair opencv-python numpy
```

### Running the Pipeline
Ensure the VisDrone dataset sequences are extracted into `VisDrone2019-MOT-val/sequences`. Execute the pipeline:
```bash
python sahi_pipeline.py
```
Processed sequence videos will automatically be saved to the `output_videos_sahi_2/` directory.

---

## Summary Report

### Base Architecture & Small Object Detection
For this drone pipeline, we selected **YOLO11 Nano** as our foundational base model. YOLO11 processes features rapidly and is highly parameter-efficient (~6 MB), satisfying the strict lightweight structural constraints. However, standard YOLO architectures traditionally compress 1080p aerial frames down to native input tensor sizes (e.g., 640x640), computationally destroying the spatial pixel features of extremely small targets. 

To overcome this, we augmented our base model with **Slicing Aided Hyper Inference (SAHI)**. The application mathematically downscales the global drone frame to exactly 720p (1280x720) and leverages SAHI to slice the image into four un-overlapped 640x640 patches. This forces YOLO11 to evaluate small persons and vehicles at near-native resolution locally in the patch, massively boosting target recall without requiring a heavier, slower AI model.

### Pipeline Performance (FPS & Hardware Tuning)
- **Test Hardware Environment:** AMD Ryzen 3 CPU (Rigorous Edge-Constraint Testing).
- **Average Pipeline FPS:** ~2.5 FPS (Global Average) | >100+ FPS (On Tracker-Only Skip Frames).

Because evaluating 4 individual SAHI AI slices sequentially on a constrained CPU is computationally heavy, we engineered an **interleaved inference cycle** (`inference_interval = 4`). The YOLO network only runs once every four frames. For the 3 intermediate frames, we completely bypass the AI and rely entirely on mathematical tracking logic (Kalman Filter state updates). This dramatically drops the CPU bottleneck, showcasing an engineering trade-off that balances AI precision with viable real-time processing simulation on low-power architectures.

### Handling "ID Switching" from Drone Ego-Motion & Occlusions
Drones introduce continuous ego-motion (panning, tilting), radically shifting the background coordinate space. This causes traditional Multi-Object Trackers to fragment bounding boxes, generating severe "ghost tracks" and dropping IDs.

We resolved this by integrating **Norfair's CPU-friendly MotionEstimator**. It runs a rapid optical flow calculation across the frame (`TranslationTransformationGetter`) to measure the exact background coordinate drift between consecutive sequences. We feed this extracted pixel shift matrix directly into the tracker, acting as **Camera Motion Compensation (CMC)**. Even during the 3 artificial frames where YOLO is skipped entirely (or when persons are briefly occluded), the CMC dynamically warps the historical bounding box coordinates to the exact changing perspective of the panning drone. This robustly anchors the IDs to the shifting ground plane and minimizes switching.

### Adapting to Edge Hardware (e.g., NVIDIA Jetson)
To fully transition this CPU-bound pipeline to dedicated GPU-accelerated edge hardware (such as an NVIDIA Jetson AGX or Orin Nano), the following adaptations would be constructed:
1. **TensorRT Optimization:** We would export the PyTorch weights (`yolo11n.pt`) to a TensorRT `.engine` context enforcing FP16/INT8 quantization. This unlocks the Jetson's Tensor Cores, pushing native detection FPS limits exponentially higher.
2. **Hardware-Accelerated SAHI:** With massive parallel GPU capacity, the CPU throttling interval could be lowered or removed entirely (`inference_interval = 1`). Furthermore, slice overlap ratios could be safely introduced to stitch target edges preventing truncation at frame seams.
3. **Appearance-Based ReID:** Instead of Norfair's lightweight spatial-distance tracking, we would deploy **DeepSORT or ByteTrack**, computing deep visual Re-Identification embeddings directly on the GPU to aggressively recover lost IDs across long-term occlusions.
4. **V4L2 / Hardware Video I/O:** The CPU-bound OpenCV VideoCapture loops would be replaced with GStreamer elements utilizing Jetson’s NVDEC/NVENC hardware encoders, nullifying I/O latency when processing active RTSP drone feeds.
