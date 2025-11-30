"""
Utility functions for the multi-camera person tracking system.

This module provides validation and calculation utilities used throughout
the application.
"""

import os
import sys
import cv2
import math
import numpy as np
from typing import List, Tuple


def validate_video_files(video_paths: List[str]) -> bool:
    """
    Check if video files exist and can be opened by OpenCV.
    
    Args:
        video_paths: List of video file paths to validate
        
    Returns:
        True if all videos are valid
        
    Raises:
        ValueError: If no video files specified
        FileNotFoundError: If any video file does not exist
        ValueError: If any video file cannot be opened or read
    """
    if not video_paths:
        raise ValueError("No video files specified")
    
    for path in video_paths:
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        
        # Try to read one frame to ensure it's a valid video
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read frames from video: {path}")
    
    return True


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


def validate_point_count(points: List[Tuple[float, float]], expected: int = 4) -> None:
    """
    Validate that we have the expected number of points.
    
    Args:
        points: List of (x, y) coordinate tuples
        expected: Expected number of points (default: 4)
        
    Raises:
        ValueError: If point count doesn't match expected
    """
    if len(points) != expected:
        raise ValueError(f"Expected {expected} points, got {len(points)}")


def validate_points_not_collinear(points: List[Tuple[float, float]]) -> None:
    """
    Validate that points are not collinear and form a valid quadrilateral.
    
    Args:
        points: List of (x, y) coordinate tuples
        
    Raises:
        ValueError: If points are too close together or collinear
    """
    # Convert points to numpy array
    points_array = np.array(points, dtype=np.float32)
    
    # Use cv2.contourArea to compute area
    area = cv2.contourArea(points_array)
    
    # Area threshold of 100 pixels² for valid quadrilateral
    if area < 100:
        raise ValueError("Points are too close or collinear (area < 100 pixels²)")


def validate_homography_matrix(matrix: np.ndarray) -> None:
    """
    Validate that homography matrix is valid and not singular.
    
    Args:
        matrix: 3x3 homography matrix
        
    Raises:
        ValueError: If matrix is None or singular
    """
    if matrix is None:
        raise ValueError("Homography computation failed")
    
    # Compute determinant
    det = np.linalg.det(matrix)
    
    # Check if matrix is singular (determinant near zero)
    if abs(det) < 1e-6:
        raise ValueError(f"Homography matrix is singular (det={det:.2e})")
