"""
Utility functions for the multi-camera person tracking system.

This module provides validation and calculation utilities used throughout
the application.
"""

import os
import cv2
import math
from typing import List, Tuple


def validate_video_files(video_paths: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if video files exist and can be opened by OpenCV.
    
    Args:
        video_paths: List of video file paths to validate
        
    Returns:
        Tuple of (all_valid, error_messages) where:
        - all_valid: True if all videos are valid, False otherwise
        - error_messages: List of error messages for invalid videos
    """
    error_messages = []
    
    if not video_paths:
        error_messages.append("No video paths provided")
        return False, error_messages
    
    for path in video_paths:
        # Check if file exists
        if not os.path.exists(path):
            error_messages.append(f"File does not exist: {path}")
            continue
        
        # Try to open with OpenCV
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                error_messages.append(f"Cannot open video file: {path}")
            else:
                # Try to read one frame to ensure it's a valid video
                ret, _ = cap.read()
                if not ret:
                    error_messages.append(f"Cannot read frames from video: {path}")
            cap.release()
        except Exception as e:
            error_messages.append(f"Error opening {path}: {str(e)}")
    
    return len(error_messages) == 0, error_messages


def calculate_grid_dimensions(num_cameras: int) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions (rows, cols) for displaying cameras.
    
    The function creates a grid that can accommodate the given number of cameras
    in a roughly square layout.
    
    Examples:
        1 camera  -> (1, 1)
        2 cameras -> (1, 2)
        3 cameras -> (2, 2)
        4 cameras -> (2, 2)
        5 cameras -> (2, 3)
        6 cameras -> (2, 3)
        9 cameras -> (3, 3)
    
    Args:
        num_cameras: Number of cameras to display
        
    Returns:
        Tuple of (rows, cols) for the grid layout
        
    Raises:
        ValueError: If num_cameras is less than 1
    """
    if num_cameras < 1:
        raise ValueError("Number of cameras must be at least 1")
    
    # Calculate the square root and round up for columns
    cols = math.ceil(math.sqrt(num_cameras))
    
    # Calculate rows needed
    rows = math.ceil(num_cameras / cols)
    
    return rows, cols


def validate_calibration_points(points: List[Tuple[float, float]]) -> Tuple[bool, str]:
    """
    Validate that calibration points form a valid quadrilateral.
    
    A valid quadrilateral requires:
    - Exactly 4 points
    - All points have valid numeric coordinates
    - Points are not collinear
    - Points form a non-degenerate quadrilateral (non-zero area)
    
    Args:
        points: List of (x, y) coordinate tuples
        
    Returns:
        Tuple of (is_valid, error_message) where:
        - is_valid: True if points form valid quadrilateral, False otherwise
        - error_message: Description of validation error, empty string if valid
    """
    # Check if we have exactly 4 points
    if len(points) != 4:
        return False, f"Expected 4 points, got {len(points)}"
    
    # Validate each point
    for i, point in enumerate(points):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            return False, f"Point {i} is not a valid (x, y) tuple"
        
        try:
            x, y = float(point[0]), float(point[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                return False, f"Point {i} contains invalid numeric values"
        except (TypeError, ValueError):
            return False, f"Point {i} contains non-numeric values"
    
    # Calculate area using the Shoelace formula
    # This helps detect degenerate cases (collinear points, zero area)
    area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    
    if area < 1e-6:  # Essentially zero area
        return False, "Points are collinear or form a degenerate quadrilateral"
    
    return True, ""
