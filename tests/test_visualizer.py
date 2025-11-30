"""
Tests for the visualizer module.
"""

import sys
from pathlib import Path

# Add parent directory to path to import visualizer module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
from visualizer import MapVisualizer


def test_visualizer_initialization():
    """Test MapVisualizer initialization."""
    canvas_width = 1200
    canvas_height = 1200
    background_color = (50, 50, 50)
    
    visualizer = MapVisualizer(canvas_width, canvas_height, background_color)
    
    assert visualizer.canvas_width == canvas_width
    assert visualizer.canvas_height == canvas_height
    assert visualizer.background_color == background_color
    assert visualizer.window_name == "2D Event Map"
    
    visualizer.close()


def test_create_blank_canvas():
    """Test blank canvas creation."""
    visualizer = MapVisualizer(1200, 1200)
    canvas = visualizer.create_blank_canvas()
    
    # Verify shape
    assert canvas.shape == (1200, 1200, 3)
    
    # Verify dtype
    assert canvas.dtype == np.uint8
    
    # Verify background color
    assert np.all(canvas == (50, 50, 50))
    
    visualizer.close()


def test_draw_grid():
    """Test grid drawing."""
    visualizer = MapVisualizer(1200, 1200)
    canvas = visualizer.create_blank_canvas()
    
    # Draw a 2x2 grid
    visualizer.draw_grid(canvas, rows=2, cols=2)
    
    # Visual verification - display for 2 seconds
    print("\nDisplaying 2x2 grid - visual verification (2 seconds)")
    visualizer.show(canvas, wait_key=2000)
    
    # Verify that grid lines were drawn (check that not all pixels are background color)
    assert not np.all(canvas == (50, 50, 50))
    
    visualizer.close()


def test_draw_detections_single_point():
    """Test drawing a single detection point."""
    visualizer = MapVisualizer(1200, 1200)
    canvas = visualizer.create_blank_canvas()
    
    # Draw point at center
    points = [(600, 600)]
    visualizer.draw_detections(canvas, points)
    
    # Visual verification
    print("\nDisplaying single point at center (2 seconds)")
    visualizer.show(canvas, wait_key=2000)
    
    # Verify that something was drawn
    assert not np.all(canvas == (50, 50, 50))
    
    visualizer.close()


def test_draw_detections_multiple_points():
    """Test drawing multiple detection points."""
    visualizer = MapVisualizer(1200, 1200)
    canvas = visualizer.create_blank_canvas()
    
    # Draw points at corners and center
    points = [
        (50, 50),      # Top-left
        (1150, 50),    # Top-right
        (1150, 1150),  # Bottom-right
        (50, 1150),    # Bottom-left
        (600, 600),    # Center
        (300, 400),    # Random position
        (800, 700)     # Random position
    ]
    visualizer.draw_detections(canvas, points)
    
    # Visual verification
    print("\nDisplaying multiple points at various positions (3 seconds)")
    visualizer.show(canvas, wait_key=3000)
    
    visualizer.close()


def test_render_frame_without_grid():
    """Test render_frame without grid."""
    visualizer = MapVisualizer(1200, 1200)
    
    points = [(600, 600), (300, 300), (900, 900)]
    canvas = visualizer.render_frame(points, show_grid=False)
    
    # Verify canvas properties
    assert canvas.shape == (1200, 1200, 3)
    assert canvas.dtype == np.uint8
    
    # Visual verification
    print("\nDisplaying render_frame without grid (2 seconds)")
    visualizer.show(canvas, wait_key=2000)
    
    visualizer.close()


def test_render_frame_with_grid():
    """Test render_frame with grid."""
    visualizer = MapVisualizer(1200, 1200)
    
    points = [(300, 300), (600, 600), (900, 300), (300, 900)]
    canvas = visualizer.render_frame(points, show_grid=True, grid_rows=2, grid_cols=2)
    
    # Verify canvas properties
    assert canvas.shape == (1200, 1200, 3)
    assert canvas.dtype == np.uint8
    
    # Visual verification
    print("\nDisplaying render_frame with 2x2 grid (3 seconds)")
    visualizer.show(canvas, wait_key=3000)
    
    visualizer.close()


def test_animation_moving_point():
    """Test animation with a moving point."""
    visualizer = MapVisualizer(1200, 1200)
    
    print("\nRunning animation - moving point across canvas (30 frames)")
    print("Watch the green dot move from left to right")
    
    # Create 30 frames of animation
    for i in range(30):
        # Calculate position (moving from left to right)
        x = int(50 + (1100 / 29) * i)
        y = 600
        points = [(x, y)]
        
        # Render and display frame
        canvas = visualizer.render_frame(points, show_grid=True, grid_rows=2, grid_cols=2)
        key = visualizer.show(canvas, wait_key=50)  # 50ms delay between frames
        
        # Allow early exit with 'q' key
        if key == ord('q'):
            print("Animation interrupted by user")
            break
    
    visualizer.close()


def test_keyboard_input():
    """Test keyboard input handling."""
    visualizer = MapVisualizer(1200, 1200)
    
    points = [(600, 600)]
    canvas = visualizer.render_frame(points)
    
    print("\nTesting keyboard input - press 'q' to quit, any other key to continue")
    print("(Auto-closes after 5 seconds if no input)")
    
    key = visualizer.show(canvas, wait_key=5000)
    
    if key == ord('q'):
        print("User pressed 'q' - quit detected")
    elif key == -1:
        print("No key pressed (timeout)")
    else:
        print(f"User pressed key with code: {key}")
    
    visualizer.close()


def test_custom_colors():
    """Test drawing with custom colors."""
    visualizer = MapVisualizer(1200, 1200, background_color=(20, 20, 20))
    canvas = visualizer.create_blank_canvas()
    
    # Draw points with different colors
    points_green = [(300, 600), (400, 600)]
    points_red = [(600, 600), (700, 600)]
    points_blue = [(900, 600), (1000, 600)]
    
    visualizer.draw_detections(canvas, points_green, color=(0, 255, 0), radius=10)
    visualizer.draw_detections(canvas, points_red, color=(0, 0, 255), radius=10)
    visualizer.draw_detections(canvas, points_blue, color=(255, 0, 0), radius=10)
    
    print("\nDisplaying points with different colors: green, red, blue (3 seconds)")
    visualizer.show(canvas, wait_key=3000)
    
    visualizer.close()


if __name__ == "__main__":
    print("Running visualizer tests...")
    print("=" * 60)
    
    # Run tests individually for visual inspection
    test_visualizer_initialization()
    print("✓ Initialization test passed")
    
    test_create_blank_canvas()
    print("✓ Blank canvas test passed")
    
    test_draw_grid()
    print("✓ Grid drawing test passed")
    
    test_draw_detections_single_point()
    print("✓ Single point detection test passed")
    
    test_draw_detections_multiple_points()
    print("✓ Multiple points detection test passed")
    
    test_render_frame_without_grid()
    print("✓ Render frame without grid test passed")
    
    test_render_frame_with_grid()
    print("✓ Render frame with grid test passed")
    
    test_animation_moving_point()
    print("✓ Animation test passed")
    
    test_keyboard_input()
    print("✓ Keyboard input test passed")
    
    test_custom_colors()
    print("✓ Custom colors test passed")
    
    print("=" * 60)
    print("All visualizer tests completed!")
