"""
Configuration file for multi-camera person tracking system.

This module contains all configuration parameters for video processing,
YOLO detection, and visualization settings.
"""

# Video source configuration
VIDEO_FILES = [
    "/home/ezio/Documents/work/vis-tracking/videos/camera1.mp4",
    "/home/ezio/Documents/work/vis-tracking/videos/camera2.mp4",
]

# Detection parameters
DETECTION_CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL = "./models/yolov8n.pt"

# Canvas/visualization settings
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1200
CANVAS_BACKGROUND_COLOR = (50, 50, 50)

# Dot rendering (for tracked persons on floor map)
DOT_COLOR = (0, 255, 0)
DOT_RADIUS = 8
DOT_THICKNESS = -1  # -1 means filled circle

# Point rendering (for calibration points)
POINT_COLOR = (0, 0, 255)
POINT_RADIUS = 5
