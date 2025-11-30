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
from utils import validate_point_count, validate_points_not_collinear, validate_homography_matrix


class GridArrangementUI:
    """Interactive UI for arranging cameras on grid layout."""
    
    def __init__(self, num_cameras, camera_frames, canvas_width, canvas_height):
        """
        Initialize grid arrangement UI.
        
        Args:
            num_cameras: int, number of cameras to arrange
            camera_frames: dict mapping camera_id -> frame (numpy array)
            canvas_width: int, canvas width in pixels
            canvas_height: int, canvas height in pixels
        """
        self.num_cameras = num_cameras
        self.camera_frames = camera_frames
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Calculate grid dimensions
        self.grid_rows, self.grid_cols = calculate_grid_dimensions(num_cameras)
        self.cell_width = canvas_width / self.grid_cols
        self.cell_height = canvas_height / self.grid_rows
        
        # Initialize default arrangement (row-major order)
        self.camera_positions = {}
        for camera_id in range(num_cameras):
            row = camera_id // self.grid_cols
            col = camera_id % self.grid_cols
            self.camera_positions[camera_id] = (row, col)
        
        self.window_name = "Camera Grid Arrangement"
    
    def _create_thumbnail(self, frame, max_width, max_height):
        """Create thumbnail of frame that fits within specified dimensions."""
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        new_w = int(w * scale * 0.8)  # 80% of available space
        new_h = int(h * scale * 0.8)
        return cv2.resize(frame, (new_w, new_h))
    
    def _draw_grid_layout(self):
        """Draw the current grid layout with camera labels and thumbnails."""
        # Create canvas
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 
                        (50, 50, 50), dtype=np.uint8)
        
        # Draw grid lines
        for i in range(1, self.grid_cols):
            x = int(i * self.cell_width)
            cv2.line(canvas, (x, 0), (x, self.canvas_height), (100, 100, 100), 2)
        
        for i in range(1, self.grid_rows):
            y = int(i * self.cell_height)
            cv2.line(canvas, (0, y), (self.canvas_width, y), (100, 100, 100), 2)
        
        # Draw cameras in their assigned grid cells
        for camera_id, (row, col) in self.camera_positions.items():
            x_start = int(col * self.cell_width)
            y_start = int(row * self.cell_height)
            x_end = int((col + 1) * self.cell_width)
            y_end = int((row + 1) * self.cell_height)
            
            # Draw cell border (highlighted)
            cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), 
                         (0, 255, 0), 3)
            
            # Draw camera label
            label = f"Camera {camera_id}"
            label_pos = (x_start + 10, y_start + 35)
            cv2.putText(canvas, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Draw thumbnail if frame available
            if camera_id in self.camera_frames:
                frame = self.camera_frames[camera_id]
                thumb_max_w = int(self.cell_width - 20)
                thumb_max_h = int(self.cell_height - 80)
                thumbnail = self._create_thumbnail(frame, thumb_max_w, thumb_max_h)
                
                # Center thumbnail in cell
                thumb_h, thumb_w = thumbnail.shape[:2]
                thumb_x = x_start + (int(self.cell_width) - thumb_w) // 2
                thumb_y = y_start + 60
                
                # Place thumbnail on canvas
                if thumb_y + thumb_h <= y_end and thumb_x + thumb_w <= x_end:
                    canvas[thumb_y:thumb_y+thumb_h, thumb_x:thumb_x+thumb_w] = thumbnail
        
        # Draw instruction box at bottom
        instruction_h = 80
        cv2.rectangle(canvas, (0, self.canvas_height - instruction_h), 
                     (self.canvas_width, self.canvas_height), (0, 0, 0), -1)
        cv2.rectangle(canvas, (0, self.canvas_height - instruction_h), 
                     (self.canvas_width, self.canvas_height), (255, 255, 255), 2)
        
        instructions = [
            "Press 'c' to CONFIRM arrangement",
            "Press 'e' to EDIT positions",
            "Press 'r' to RESET to default"
        ]
        
        for i, text in enumerate(instructions):
            y_pos = self.canvas_height - instruction_h + 25 + i * 20
            cv2.putText(canvas, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return canvas
    
    def _parse_arrangement_input(self, input_str):
        """
        Parse user input for camera arrangement.
        
        Format: "camera_id:row,col camera_id:row,col ..."
        Example: "0:0,1 1:0,0 2:1,0"
        
        Returns:
            dict mapping camera_id -> (row, col), or None if invalid
        """
        try:
            new_positions = {}
            parts = input_str.strip().split()
            
            for part in parts:
                if ':' not in part or ',' not in part:
                    return None
                
                camera_part, pos_part = part.split(':')
                row_str, col_str = pos_part.split(',')
                
                camera_id = int(camera_part)
                row = int(row_str)
                col = int(col_str)
                
                # Validate camera_id
                if camera_id < 0 or camera_id >= self.num_cameras:
                    print(f"  Error: Invalid camera_id {camera_id}")
                    return None
                
                # Validate grid position
                if row < 0 or row >= self.grid_rows:
                    print(f"  Error: Row {row} out of range (0-{self.grid_rows-1})")
                    return None
                
                if col < 0 or col >= self.grid_cols:
                    print(f"  Error: Col {col} out of range (0-{self.grid_cols-1})")
                    return None
                
                new_positions[camera_id] = (row, col)
            
            # Validate all cameras are assigned
            if len(new_positions) != self.num_cameras:
                print(f"  Error: Must specify all {self.num_cameras} cameras")
                return None
            
            # Check for duplicate positions
            position_set = set(new_positions.values())
            if len(position_set) != len(new_positions):
                print(f"  Error: Duplicate grid positions (cameras cannot overlap)")
                return None
            
            return new_positions
            
        except Exception as e:
            print(f"  Error parsing input: {e}")
            return None
    
    def _edit_arrangement(self):
        """Allow user to manually edit camera positions via text input."""
        print("\n=== Edit Camera Arrangement ===")
        print(f"Grid size: {self.grid_rows} rows x {self.grid_cols} cols")
        print("Current arrangement:")
        for camera_id in sorted(self.camera_positions.keys()):
            row, col = self.camera_positions[camera_id]
            print(f"  Camera {camera_id}: row {row}, col {col}")
        
        print("\nEnter new arrangement:")
        print("Format: camera_id:row,col camera_id:row,col ...")
        print(f"Example: 0:0,1 1:0,0 2:1,0")
        print("(Rows: 0-{}, Cols: 0-{})".format(self.grid_rows-1, self.grid_cols-1))
        
        input_str = input("\nNew arrangement: ").strip()
        
        if not input_str:
            print("  No changes made")
            return False
        
        new_positions = self._parse_arrangement_input(input_str)
        
        if new_positions is None:
            print("  Invalid input. Keeping current arrangement.")
            return False
        
        # Update positions
        self.camera_positions = new_positions
        print("  ✓ Arrangement updated")
        return True
    
    def arrange_cameras_simple(self):
        """
        Display current arrangement and allow user to confirm or edit.
        
        Returns:
            dict mapping camera_id -> (grid_row, grid_col)
        """
        print("\n=== Camera Grid Arrangement ===")
        print(f"Grid: {self.grid_rows} rows x {self.grid_cols} cols")
        print("Default arrangement (row-major order):")
        for camera_id in sorted(self.camera_positions.keys()):
            row, col = self.camera_positions[camera_id]
            print(f"  Camera {camera_id} -> Grid cell ({row}, {col})")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while True:
            # Draw current layout
            canvas = self._draw_grid_layout()
            cv2.imshow(self.window_name, canvas)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Confirm arrangement
                print("  ✓ Arrangement confirmed")
                cv2.destroyWindow(self.window_name)
                return self.camera_positions
            
            elif key == ord('e'):
                # Edit arrangement
                cv2.destroyWindow(self.window_name)
                
                if self._edit_arrangement():
                    # Show updated arrangement
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    print("\nUpdated arrangement displayed. Press 'c' to confirm, 'e' to edit again")
                else:
                    # Reopen window to show unchanged arrangement
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            elif key == ord('r'):
                # Reset to default
                print("  Resetting to default arrangement...")
                for camera_id in range(self.num_cameras):
                    row = camera_id // self.grid_cols
                    col = camera_id % self.grid_cols
                    self.camera_positions[camera_id] = (row, col)
                print("  ✓ Reset complete")


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
            # Check if window was closed by user
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError(f"User cancelled calibration at camera {self.camera_id}")
            
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
                    # Validate points
                    try:
                        validate_point_count(self.points, 4)
                        validate_points_not_collinear(self.points)
                        print(f"  ✓ 4 points confirmed for camera {self.camera_id}")
                        cv2.destroyWindow(self.window_name)
                        return self.points
                    except ValueError as e:
                        print(f"  Error: {e}")
                        print("  Press 'r' to reset and try again")
                else:
                    print(f"  Error: Need exactly 4 points (have {len(self.points)})")
            
            elif key == ord('q'):
                # Quit
                cv2.destroyWindow(self.window_name)
                raise RuntimeError(f"User cancelled calibration at camera {self.camera_id}")


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
    
    try:
        # Compute perspective transform
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Validate matrix
        validate_homography_matrix(H)
        
        return H
    except ValueError as e:
        raise RuntimeError(f"Homography computation failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Homography computation failed: {e}")


def calibrate_cameras(video_files, canvas_width, canvas_height):
    """
    Main calibration function for all cameras.
    
    Performs the complete calibration workflow:
    1. Loads first frame from each video
    2. User selects 4 floor boundary points for each camera
    3. Interactive grid arrangement (user confirms or customizes layout)
    4. Computes destination points based on final grid arrangement
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
    
    # Phase 1: Collect source points and frames from all cameras
    camera_source_points = {}
    camera_frames = {}
    
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
        
        # Store frame for grid arrangement UI
        camera_frames[camera_id] = frame
        
        # Run point selection
        selector = PointSelector(frame, camera_id)
        try:
            source_points = selector.select_points()
            camera_source_points[camera_id] = source_points
        except RuntimeError as e:
            # Re-raise with camera context
            raise RuntimeError(f"Camera {camera_id} point selection failed: {e}")
    
    # Phase 2: Interactive grid arrangement
    print("\n" + "="*60)
    print("PHASE 2: Camera Grid Arrangement")
    print("="*60)
    
    arrangement_ui = GridArrangementUI(
        num_cameras,
        camera_frames,
        canvas_width,
        canvas_height
    )
    
    # Get final camera positions (user confirms or customizes)
    camera_positions = arrangement_ui.arrange_cameras_simple()
    
    # Calculate cell dimensions
    cell_width = canvas_width / grid_cols
    cell_height = canvas_height / grid_rows
    
    # Phase 3: Compute homographies based on final arrangement
    print("\n" + "="*60)
    print("PHASE 3: Computing Homography Matrices")
    print("="*60)
    
    cameras = []
    
    for camera_id in range(num_cameras):
        grid_row, grid_col = camera_positions[camera_id]
        source_points = camera_source_points[camera_id]
        video_file = video_files[camera_id]
        
        # Calculate destination points for assigned grid cell
        # Points in order: top_left, top_right, bottom_right, bottom_left
        top_left = (grid_col * cell_width, grid_row * cell_height)
        top_right = ((grid_col + 1) * cell_width, grid_row * cell_height)
        bottom_right = ((grid_col + 1) * cell_width, (grid_row + 1) * cell_height)
        bottom_left = (grid_col * cell_width, (grid_row + 1) * cell_height)
        
        destination_points = [top_left, top_right, bottom_right, bottom_left]
        
        # Compute homography
        try:
            homography_matrix = compute_homography(source_points, destination_points)
        except RuntimeError as e:
            # Re-raise with camera context
            raise RuntimeError(f"Camera {camera_id} homography computation failed: {e}")
        
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
