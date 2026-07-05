import cv2
import time
import numpy as np
import os
import glob
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from norfair import Tracker, Detection
from norfair.camera_motion import MotionEstimator, HomographyTransformationGetter

def sahi_to_norfair(sahi_predictions):
    """
    Converts SAHI ObjectPrediction list into Norfair Detection format.
    Filters to standard target classes: Persons (0), Bicycles (1), Cars (2), Motorbikes (3), Buses (5), Trucks (7).
    """
    norfair_detections = []
    target_classes = [0, 1, 2, 3, 5, 7]
    
    for pred in sahi_predictions:
        class_id = pred.category.id
        if class_id not in target_classes:
            continue
            
        x1, y1, x2, y2 = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        score = pred.score.value
        
        # Pass the bbox explicitly in data so we can determine proportions later
        bbox = (x1, y1, x2, y2)
        points = np.array([[center_x, center_y]])
        norfair_detections.append(
            Detection(points=points, scores=np.array([score]), data={'bbox': bbox, 'class_id': class_id})
        )
    return norfair_detections

# 1. Initialize SAHI AutoDetectionModel
print("Loading YOLO11 model into SAHI...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolo11n.onnx', 
    confidence_threshold=0.45, # Increased to filter out false detections
    device="cpu" 
)

visdrone_seq_dir = "VisDrone2019-MOT-val/sequences"
output_dir = "output_videos_sahi_2"
os.makedirs(output_dir, exist_ok=True)

for seq_name in sorted(os.listdir(visdrone_seq_dir)):
    seq_path = os.path.join(visdrone_seq_dir, seq_name)
    if not os.path.isdir(seq_path):
        continue
        
    frames = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
    if not frames:
        continue
        
    print(f"Processing sequence: {seq_name}")
    
    # Use first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    # Downscale global video to 720p to reduce number of SAHI slices to exactly 4!
    width, height = 1280, 720
    fps_input = 30 # Default assumed logical fps for sequences
    
    out_path = os.path.join(output_dir, f'sahi_output_{seq_name}.mp4')
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_input, (width, height))
    
    # Initialize Norfair Tracker.
    tracker = Tracker(
        distance_function="mean_euclidean", 
        distance_threshold=85, # Toned down slightly so it doesn't snap to distant false positives
        hit_counter_max=30,    # Halved to 30 frames (~1 sec) so ghost tracks die off reasonably fast!
        past_detections_length=10 # Reduced so it doesn't over-remember dead tracks
    )
    
    # Homography handles both ROTATION and translation of the drone camera!
    motion_estimator = MotionEstimator(
        max_points=500, min_distance=15, transformations_getter=HomographyTransformationGetter()
    )
    
    # Dictionary to store the history of tracked centers for the "tail"
    track_history = defaultdict(lambda: [])
    max_tail_length = 15 # Reduced tail length to prevent excessive clutter
    
    # Analytics 1: Persistent Heatmap Array (720p)
    heatmap_layer = np.zeros((720, 1280), dtype=np.float32)
    
    # ENGINEERING TRADEOFF: Interleaved Inference
    inference_interval = 4
    
    for idx, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        if idx % 5 == 0:
            print(f"  -> Processing frame {idx+1}/{len(frames)}")
            
        start_time = time.time()
        
        # Resize to exactly 1280x720. 
        # This allows 640x640 slices to fit smoothly (exactly 4 patches) minimizing CPU overload!
        frame_resized = cv2.resize(frame, (1280, 720))
        
        # 2.5 Camera Motion Compensation (CMC)
        # This solves the drone auto-panning issue! It calculates global background pixel shift on the CPU natively
        coord_transformations = motion_estimator.update(frame_resized)

        # 3. SAHI Sliced Inference
        if idx % inference_interval == 0:
            result = get_sliced_prediction(
                frame_resized,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.0,
                overlap_width_ratio=0.0,
                verbose=0
            )
            sahi_predictions = result.object_prediction_list
            detections = sahi_to_norfair(sahi_predictions)
            tracked_objects = tracker.update(detections=detections, coord_transformations=coord_transformations)
        else:
            detections = []
            tracked_objects = tracker.update(detections=detections, coord_transformations=coord_transformations)
            # Artificial Life Support: Prevent Norfair from killing objects during skipped CPU frames!
            for obj in tracker.tracked_objects:
                # Offset the -1 penalty of missing this non-inference frame to keep the Kalman track alive
                obj.hit_counter += 1

        # 5. Processing Results & Drawing
        annotated_frame = frame_resized.copy()
        
        for tracked_obj in tracked_objects:
            
            # Do not draw if it is a heavily hit-missed or dead tracking id
            if tracked_obj.last_detection is None:
                continue
            
            track_id = tracked_obj.id
            
            # Retrieve the underlying customized data proportions to draw properly
            bbox = tracked_obj.last_detection.data['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Utilize Norfair's Kalman Filter estimate for smooth movement logic through occlusion/gaps
            center_x, center_y = tracked_obj.estimate[0]
            x1 = int(center_x - w/2)
            y1 = int(center_y - h/2)
            x2 = int(center_x + w/2)
            y2 = int(center_y + h/2)
            
            center_x, center_y = int(center_x), int(center_y)

            # Analytics 2: Speed Estimation (Pixels per second)
            # Calculate speed based on the pixel distance from the last known center
            track = track_history[track_id]
            if len(track) > 0:
                last_x, last_y = track[-1]
                vx = center_x - last_x
                vy = center_y - last_y
                speed_px_per_frame = np.sqrt(vx**2 + vy**2)
                speed_px_per_sec = speed_px_per_frame * fps_input
            else:
                speed_px_per_frame = 0.0
                speed_px_per_sec = 0.0

            # Draw Bounding Box, ID, and Speed
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Pink SAHI boxes
            label = f"ID: {track_id} | {speed_px_per_sec:.0f} px/s"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Analytics 1: Add this object's center to the heatmap
            cv2.circle(heatmap_layer, (center_x, center_y), radius=12, color=255.0, thickness=-1)

            # If the tracking line jumps impossibly fast (tracker glitch / ID switch), reset the tail!
            if speed_px_per_frame > 150:
                track.clear()

            # Update and Draw Trajectory Tail logically using the smoothed centers
            track.append((center_x, center_y))
            if len(track) > max_tail_length:
                track.pop(0)

            # Draw the tail using polylines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

        # Analytics 1: Blend Heatmap Layer onto Annotated Frame
        # Fade the heatmap quickly so old traffic cools down and disappears immediately!
        heatmap_layer *= 0.80
        
        # Normalize heatmap to 0-255 scale
        heatmap_norm = np.clip(heatmap_layer, 0, 255).astype(np.uint8)
        
        # Apply Jet colormap (Blue=Cold, Red=Hot)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Create a mask so we only tint the areas where vehicles have actually driven
        active_mask = heatmap_norm > 0
        
        # Blend manually using numpy to avoid OpenCV shape mismatch
        annotated_frame[active_mask] = (
            annotated_frame[active_mask] * 0.6 + heatmap_color[active_mask] * 0.4
        ).astype(np.uint8)

        # 6. FPS Calculation & Hardware Logging
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        cv2.putText(annotated_frame, f"CPU FPS: {fps:.1f} (Interleaved SAHI + Norfair)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Architecture: 720p SAHI Slicing (1 in {inference_interval} frames) + Kalman Smoothing", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Write and display
        out.write(annotated_frame)
        
        # Real-time visual feedback
        cv2.imshow("SAHI Tracking", cv2.resize(annotated_frame, (800, 600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Sequence processing stopped early by user.")
            break

    out.release()
    print(f"Finished sequence: {seq_name}")

cv2.destroyAllWindows()
