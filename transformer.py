"""
Transformer module for multi-camera person tracking system.

This module handles perspective transformations to map detected person positions
from camera image coordinates to real-world floor map coordinates. It uses
homography matrices derived from calibration points to perform the transformation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def transform_point(point: Tuple[float, float], homography_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Transform a single point using a homography matrix.
    
    Args:
        point: (x, y) tuple in source coordinate system
        homography_matrix: 3x3 homography transformation matrix
        
    Returns:
        Transformed (x, y) tuple in destination coordinate system
    """
    # Format point for cv2.perspectiveTransform: shape must be (1, 1, 2)
    point_array = np.array([[point]], dtype=np.float32)
    
    # Apply perspective transformation
    transformed = cv2.perspectiveTransform(point_array, homography_matrix)
    
    # Extract and return as tuple
    return (float(transformed[0][0][0]), float(transformed[0][0][1]))


def transform_points(points: List[Tuple[float, float]], homography_matrix: np.ndarray) -> List[Tuple[float, float]]:
    """
    Transform multiple points using a homography matrix in batch.
    
    Args:
        points: List of (x, y) tuples in source coordinate system
        homography_matrix: 3x3 homography transformation matrix
        
    Returns:
        List of transformed (x, y) tuples in destination coordinate system
    """
    # Handle empty list
    if not points:
        return []
    
    # Format points for cv2.perspectiveTransform: shape must be (1, N, 2)
    points_array = np.array([points], dtype=np.float32)
    
    # Apply perspective transformation
    transformed = cv2.perspectiveTransform(points_array, homography_matrix)
    
    # Convert back to list of tuples
    return [(float(x), float(y)) for x, y in transformed[0]]


def validate_canvas_coordinates(
    points: List[Tuple[float, float]], 
    canvas_width: int, 
    canvas_height: int
) -> Tuple[List[Tuple[float, float]], int]:
    """
    Filter points to only those within canvas bounds.
    
    Args:
        points: List of (x, y) tuples to validate
        canvas_width: Width of the canvas
        canvas_height: Height of the canvas
        
    Returns:
        Tuple of (valid_points, invalid_count) where:
        - valid_points: List of points within bounds (0 <= x <= width, 0 <= y <= height)
        - invalid_count: Number of out-of-bounds points
    """
    valid_points = []
    invalid_count = 0
    
    for x, y in points:
        if 0 <= x <= canvas_width and 0 <= y <= canvas_height:
            valid_points.append((x, y))
        else:
            invalid_count += 1
    
    return valid_points, invalid_count


class CoordinateTransformer:
    """
    Transforms detected person positions from camera coordinates to canvas coordinates.
    
    This class manages the transformation of foot positions detected in multiple camera
    views to a unified canvas coordinate system using homography matrices from calibration.
    """
    
    def __init__(self, calibration_data: Dict[str, Any]):
        """
        Initialize the coordinate transformer with calibration data.
        
        Args:
            calibration_data: Dictionary containing:
                - canvas_width: Width of the canvas in pixels
                - canvas_height: Height of the canvas in pixels
                - cameras: List of camera calibration data, each containing:
                    - camera_id: Unique identifier for the camera
                    - homography_matrix: 3x3 transformation matrix
        """
        # Extract canvas dimensions
        self.canvas_width = calibration_data['canvas_width']
        self.canvas_height = calibration_data['canvas_height']
        
        # Build dictionary mapping camera_id -> homography_matrix
        self.camera_homographies = {}
        for camera in calibration_data['cameras']:
            camera_id = camera['camera_id']
            homography = np.array(camera['homography_matrix'], dtype=np.float32)
            self.camera_homographies[camera_id] = homography
    
    def transform_detections(self, detections_per_camera: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[float, float]]:
        """
        Transform detections from all cameras to canvas coordinates.
        
        Args:
            detections_per_camera: Dictionary mapping camera_id -> list of detection dicts.
                Each detection dict must contain 'foot_position': (x, y) tuple.
                
        Returns:
            List of (x, y) tuples representing valid positions on canvas.
            Points outside canvas bounds are filtered out.
        """
        all_canvas_positions = []
        
        # Process each camera's detections
        for camera_id, detections in detections_per_camera.items():
            # Skip if camera not calibrated
            if camera_id not in self.camera_homographies:
                continue
            
            # Extract foot positions from detections
            foot_positions = [detection['foot_position'] for detection in detections]
            
            # Skip if no detections
            if not foot_positions:
                continue
            
            # Get homography matrix for this camera
            homography = self.camera_homographies[camera_id]
            
            # Transform all foot positions to canvas coordinates
            transformed_points = transform_points(foot_positions, homography)
            
            # Validate that points are within canvas bounds
            valid_points, invalid_count = validate_canvas_coordinates(
                transformed_points, 
                self.canvas_width, 
                self.canvas_height
            )
            
            # Aggregate valid points
            all_canvas_positions.extend(valid_points)
        
        return all_canvas_positions
