"""
Detector module for multi-camera person tracking system.

This module handles person detection using YOLO (You Only Look Once) models.
It processes video frames to detect persons and extracts their bounding boxes
and foot positions for tracking and coordinate transformation.
"""

import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Union


class PersonDetector:
    """
    YOLOv8-based person detector.
    
    Detects persons in frames and extracts bounding boxes and foot positions
    for downstream 3D tracking.
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the person detector.
        
        Args:
            model_name: YOLOv8 model weights filename (default: yolov8n.pt)
            confidence_threshold: Minimum confidence for valid detections (0.0-1.0)
        
        Raises:
            RuntimeError: If model fails to load
        """
        try:
            self.model = YOLO(model_name)
            self.confidence_threshold = confidence_threshold
            self.person_class_id = 0  # COCO dataset person class ID
            print(f"[PersonDetector] Loaded model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {e}")
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in a single frame.
        
        Args:
            frame: Input image frame (numpy array, BGR format)
        
        Returns:
            List of detection dictionaries with structure:
            [
                {
                    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
                    'confidence': float,        # Detection confidence score
                    'foot_position': (x, y)     # Bottom-center point (for 3D tracking)
                },
                ...
            ]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            # Extract person detections from results
            if len(results) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get class ID and check if it's a person
                    class_id = int(box.cls.cpu().numpy()[0])
                    
                    if class_id != self.person_class_id:
                        continue
                    
                    # Get confidence and filter by threshold
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Extract bounding box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, xyxy)
                    
                    # Calculate foot position (bottom-center of bounding box)
                    center_x = (x1 + x2) / 2
                    bottom_y = y2
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'foot_position': (center_x, bottom_y)
                    })
            
            return detections
            
        except Exception as e:
            print(f"[PersonDetector] Error during detection: {e}")
            return []
    
    def detect_persons_batch(self, frames: Union[List[np.ndarray], Dict[str, np.ndarray]]) -> Union[List[List[Dict]], Dict[str, List[Dict]]]:
        """
        Detect persons in multiple frames.
        
        Args:
            frames: List of frames or dictionary mapping camera IDs to frames
        
        Returns:
            Corresponding list or dictionary of detection lists
        """
        if isinstance(frames, dict):
            # Process dictionary of frames (camera_id -> frame)
            return {
                camera_id: self.detect_persons(frame)
                for camera_id, frame in frames.items()
            }
        else:
            # Process list of frames
            return [
                self.detect_persons(frame)
                for frame in frames
            ]
