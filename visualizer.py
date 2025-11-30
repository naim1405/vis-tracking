"""
Visualizer module for multi-camera person tracking system.

This module handles all visualization tasks including:
- Displaying camera feeds in a grid layout
- Rendering the 2D floor map with detected person positions
- Drawing calibration points during setup
- Updating the display in real-time during tracking
"""

import numpy as np
import cv2
from typing import List, Tuple


class MapVisualizer:
    """Visualizes detection points on a 2D canvas map."""
    
    def __init__(self, canvas_width: int, canvas_height: int, background_color: Tuple[int, int, int] = (50, 50, 50)):
        """
        Initialize the map visualizer.
        
        Args:
            canvas_width: Width of the canvas in pixels
            canvas_height: Height of the canvas in pixels
            background_color: BGR color tuple for the canvas background
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.background_color = background_color
        self.window_name = "2D Event Map"
        
        # Create and configure OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, canvas_width, canvas_height)
    
    def create_blank_canvas(self) -> np.ndarray:
        """
        Create a blank canvas filled with the background color.
        
        Returns:
            Canvas as numpy array with shape (height, width, 3)
        """
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 
                        self.background_color, dtype=np.uint8)
        return canvas
    
    def draw_grid(self, canvas: np.ndarray, rows: int, cols: int, 
                  color: Tuple[int, int, int] = (80, 80, 80)) -> None:
        """
        Draw grid lines on the canvas.
        
        Args:
            canvas: Canvas to draw on
            rows: Number of grid rows
            cols: Number of grid columns
            color: BGR color tuple for grid lines
        """
        height, width = canvas.shape[:2]
        
        # Draw vertical lines
        col_width = width / cols
        for i in range(1, cols):
            x = int(i * col_width)
            cv2.line(canvas, (x, 0), (x, height), color, 1)
        
        # Draw horizontal lines
        row_height = height / rows
        for i in range(1, rows):
            y = int(i * row_height)
            cv2.line(canvas, (0, y), (width, y), color, 1)
    
    def draw_detections(self, canvas: np.ndarray, points: List[Tuple[float, float]], 
                       color: Tuple[int, int, int] = (0, 255, 0), 
                       radius: int = 8, thickness: int = -1) -> None:
        """
        Draw detection points on the canvas.
        
        Args:
            canvas: Canvas to draw on
            points: List of (x, y) coordinate tuples
            color: BGR color tuple for the circles
            radius: Radius of the circles in pixels
            thickness: Circle thickness (-1 for filled)
        """
        for point in points:
            # Convert to integer coordinates
            center = (int(point[0]), int(point[1]))
            cv2.circle(canvas, center, radius, color, thickness)
    
    def render_frame(self, detection_points: List[Tuple[float, float]], 
                    show_grid: bool = False, grid_rows: int = 2, 
                    grid_cols: int = 2) -> np.ndarray:
        """
        Render a complete frame with optional grid and detection points.
        
        Args:
            detection_points: List of (x, y) detection coordinates
            show_grid: Whether to draw grid lines
            grid_rows: Number of grid rows (if show_grid is True)
            grid_cols: Number of grid columns (if show_grid is True)
            
        Returns:
            Rendered canvas as numpy array
        """
        # Create blank canvas
        canvas = self.create_blank_canvas()
        
        # Optionally draw grid
        if show_grid:
            self.draw_grid(canvas, grid_rows, grid_cols)
        
        # Draw detection points
        self.draw_detections(canvas, detection_points)
        
        return canvas
    
    def show(self, canvas: np.ndarray, wait_key: int = 1) -> int:
        """
        Display the canvas in the window.
        
        Args:
            canvas: Canvas to display
            wait_key: Milliseconds to wait for key press (0 = wait indefinitely)
            
        Returns:
            Key code of pressed key, or -1 if no key was pressed
        """
        cv2.imshow(self.window_name, canvas)
        return cv2.waitKey(wait_key)
    
    def close(self) -> None:
        """Close the visualization window."""
        cv2.destroyWindow(self.window_name)
