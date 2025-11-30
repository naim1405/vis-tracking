"""
Calibration module for multi-camera person tracking system.

This module handles the calibration process for mapping 2D image coordinates
from camera views to 2D floor plan coordinates. Users select four corresponding
points in each camera view and on the floor map to establish perspective
transformations.
"""

import cv2
import numpy as np
import math


class PointSelector:
    """Interactive point selection for floor boundary calibration."""
    
    def __init__(self, frame, camera_id):
        """
        Initialize point selector with camera frame.
        
        Args:
            frame: numpy array (BGR image)
            camera_id: int identifier for the camera
        """
        self.frame = frame.copy()
        self.camera_id = camera_id
        self.points = []
        self.window_name = f"Camera {camera_id} - Select 4 Floor Corners"
        self.display_frame = None
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"  Point {len(self.points)}/4: ({x}, {y})")
    
    def _draw_points(self):
        """Draw selected points and connecting lines on display frame."""
        self.display_frame = self.frame.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(self.display_frame, point, 5, (0, 0, 255), -1)
            # Add point number label
            cv2.putText(self.display_frame, str(i + 1), 
                       (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw lines connecting points
        if len(self.points) > 1:
            for i in range(len(self.points)):
                start = self.points[i]
                end = self.points[(i + 1) % len(self.points)]
                if i < len(self.points) - 1 or len(self.points) == 4:
                    cv2.line(self.display_frame, start, end, (0, 255, 0), 2)
        
        # Draw instruction text
        point_count = f"Points: {len(self.points)}/4"
        instructions = "r=reset | c=confirm (4 pts) | q=quit"
        
        # Background rectangle for text
        cv2.rectangle(self.display_frame, (10, 10), (450, 80), 
                     (0, 0, 0), -1)
        cv2.rectangle(self.display_frame, (10, 10), (450, 80), 
                     (255, 255, 255), 2)
        
        cv2.putText(self.display_frame, point_count,
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 255, 255), 2)
        cv2.putText(self.display_frame, instructions,
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (255, 255, 255), 1)
    
    def select_points(self):
        """
        Display frame and allow user to select 4 floor boundary points.
        
        Returns:
            list of 4 (x, y) tuples representing floor corners
            
        Raises:
            RuntimeError: if user quits or invalid points selected
        """
        print(f"\n=== Camera {self.camera_id} Calibration ===")
        print("Click 4 corners of the floor area (in order)")
        print("Press 'r' to reset, 'c' to confirm (when 4 points), 'q' to quit")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        while True:
            self._draw_points()
            cv2.imshow(self.window_name, self.display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                # Reset points
                print("  Points reset")
                self.points = []
            
            elif key == ord('c'):
                # Confirm selection
                if len(self.points) == 4:
                    # Validate points form valid quadrilateral
                    if self._validate_points():
                        print(f"  ✓ 4 points confirmed for camera {self.camera_id}")
                        cv2.destroyWindow(self.window_name)
                        return self.points
                    else:
                        print("  Error: Points are invalid (too close or collinear)")
                        print("  Press 'r' to reset and try again")
                else:
                    print(f"  Error: Need exactly 4 points (have {len(self.points)})")
            
            elif key == ord('q'):
                # Quit
                cv2.destroyWindow(self.window_name)
                raise RuntimeError(f"User cancelled calibration at camera {self.camera_id}")
        
    def _validate_points(self):
        """
        Validate that points form a valid quadrilateral.
        
        Returns:
            bool: True if points are valid
        """
        if len(self.points) != 4:
            return False
        
        # Check area is not too small (points not collinear/too close)
        points_array = np.array(self.points, dtype=np.float32)
        area = cv2.contourArea(points_array)
        
        # Area threshold (adjust based on expected frame size)
        min_area = 100
        return area > min_area


def calculate_grid_dimensions(num_cameras):
    """
    Calculate grid layout dimensions for arranging cameras.
    
    Args:
        num_cameras: number of cameras to arrange
        
    Returns:
        tuple: (rows, cols) for grid layout
    """
    # Formula: rows = ceil(sqrt(n)), cols = ceil(n/rows)
    # This creates a roughly square grid
    rows = math.ceil(math.sqrt(num_cameras))
    cols = math.ceil(num_cameras / rows)
    return rows, cols


def compute_homography(source_points, destination_points):
    """
    Compute homography matrix from source to destination points.
    
    Args:
        source_points: list of 4 (x, y) tuples in camera coordinates
        destination_points: list of 4 (x, y) tuples in canvas coordinates
        
    Returns:
        numpy array: 3x3 homography matrix
        
    Raises:
        RuntimeError: if homography computation fails or matrix is invalid
    """
    # Convert to numpy arrays with correct shape for OpenCV
    src_pts = np.array(source_points, dtype=np.float32)
    dst_pts = np.array(destination_points, dtype=np.float32)
    
    # Compute perspective transform
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    if H is None:
        raise RuntimeError("Failed to compute homography matrix")
    
    # Validate matrix is not singular
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        raise RuntimeError(f"Homography matrix is singular (det={det})")
    
    return H


def calibrate_cameras(video_files, canvas_width, canvas_height):
    """
    Main calibration function for all cameras.
    
    Performs the complete calibration workflow:
    1. Loads first frame from each video
    2. User selects 4 floor boundary points for each camera
    3. Calculates grid layout for camera arrangement
    4. Computes destination points based on grid cells
    5. Computes homography matrices
    
    Args:
        video_files: list of video file paths
        canvas_width: width of canvas in pixels
        canvas_height: height of canvas in pixels
        
    Returns:
        dict: calibration data with structure:
        {
            'canvas_width': int,
            'canvas_height': int,
            'cameras': [
                {
                    'camera_id': int,
                    'video_file': str,
                    'source_points': [(x,y), ...],
                    'destination_points': [(x,y), ...],
                    'homography_matrix': np.array,
                    'grid_cell': (row, col)
                },
                ...
            ]
        }
        
    Raises:
        ValueError: if video files cannot be opened
        RuntimeError: if calibration fails for any camera
    """
    num_cameras = len(video_files)
    print(f"\nCalibrating {num_cameras} cameras...")
    
    # Calculate grid layout
    grid_rows, grid_cols = calculate_grid_dimensions(num_cameras)
    print(f"Grid layout: {grid_rows}x{grid_cols}")
    
    # Calculate cell dimensions
    cell_width = canvas_width / grid_cols
    cell_height = canvas_height / grid_rows
    
    cameras = []
    
    # Process each camera
    for camera_id, video_file in enumerate(video_files):
        print(f"\n--- Camera {camera_id}: {video_file} ---")
        
        # Load first frame
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_file}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read first frame from: {video_file}")
        
        # Run point selection
        selector = PointSelector(frame, camera_id)
        try:
            source_points = selector.select_points()
        except Exception as e:
            raise RuntimeError(f"Point selection failed for camera {camera_id}: {e}")
        
        # Calculate grid cell position (row-major order)
        grid_row = camera_id // grid_cols
        grid_col = camera_id % grid_cols
        
        # Calculate destination points for this grid cell
        # Points in order: top_left, top_right, bottom_right, bottom_left
        top_left = (grid_col * cell_width, grid_row * cell_height)
        top_right = ((grid_col + 1) * cell_width, grid_row * cell_height)
        bottom_right = ((grid_col + 1) * cell_width, (grid_row + 1) * cell_height)
        bottom_left = (grid_col * cell_width, (grid_row + 1) * cell_height)
        
        destination_points = [top_left, top_right, bottom_right, bottom_left]
        
        # Compute homography
        try:
            homography_matrix = compute_homography(source_points, destination_points)
        except Exception as e:
            raise RuntimeError(f"Homography computation failed for camera {camera_id}: {e}")
        
        # Store calibration data
        camera_data = {
            'camera_id': camera_id,
            'video_file': video_file,
            'source_points': source_points,
            'destination_points': destination_points,
            'homography_matrix': homography_matrix,
            'grid_cell': (grid_row, grid_col)
        }
        cameras.append(camera_data)
        
        print(f"✓ Camera {camera_id} calibrated")
        print(f"  Grid cell: ({grid_row}, {grid_col})")
        print(f"  Source points: {source_points}")
        print(f"  Destination: [{int(top_left[0])},{int(top_left[1])}] to [{int(bottom_right[0])},{int(bottom_right[1])}]")
    
    # Build calibration data structure
    calibration_data = {
        'canvas_width': canvas_width,
        'canvas_height': canvas_height,
        'grid_layout': (grid_rows, grid_cols),
        'cameras': cameras
    }
    
    print(f"\n✓ All {num_cameras} cameras calibrated successfully")
    
    return calibration_data
