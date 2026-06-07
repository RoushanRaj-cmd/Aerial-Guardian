import cv2
import time
import numpy as np
import os
import glob
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from norfair import Tracker, Detection
from norfair.camera_motion import MotionEstimator, TranslationTransformationGetter

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
    model_type='yolov8', # 'yolov8' supports ultralytics yolo11 weights internally backward-compatibly
    model_path='yolo11n.pt',
    confidence_threshold=0.3,
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
    # Distance function 'euclidean' acts as a simple, fast tracking mechanism measuring point drift. 
    # Works excellently for low-fps aerial camera movements on limited-scale centers.
    tracker = Tracker(distance_function="euclidean", distance_threshold=75)
    
    # Initialize Camera Motion Estimator! Very critical for drones to stop Kalman "ghost" drifting!
    motion_estimator = MotionEstimator(
        max_points=500, min_distance=15, transformations_getter=TranslationTransformationGetter()
    )
    
    # Dictionary to store the history of tracked centers for the "tail"
    track_history = defaultdict(lambda: [])
    max_tail_length = 30 # Number of frames the trajectory tail will persist
    
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

            # Draw Bounding Box and ID
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Pink SAHI boxes
            cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Update and Draw Trajectory Tail logically using the smoothed centers
            track = track_history[track_id]
            track.append((center_x, center_y))
            if len(track) > max_tail_length:
                track.pop(0)

            # Draw the tail using polylines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

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
