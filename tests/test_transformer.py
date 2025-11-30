"""
Test suite for transformer module.

Tests coordinate transformation from camera coordinates to canvas coordinates
using homography matrices.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import (
    transform_point,
    transform_points,
    validate_canvas_coordinates,
    CoordinateTransformer
)


def test_transform_point_identity():
    """Test transform_point with identity matrix (should return same point)."""
    print("\n=== Testing transform_point with identity matrix ===")
    
    # Identity matrix: no transformation
    identity_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Test several points
    test_points = [(100, 200), (0, 0), (640, 480), (320.5, 240.7)]
    
    for point in test_points:
        transformed = transform_point(point, identity_matrix)
        print(f"  Input: {point} -> Output: {transformed}")
        
        # With identity matrix, output should equal input (within floating point precision)
        assert abs(transformed[0] - point[0]) < 0.01, f"X mismatch: {transformed[0]} != {point[0]}"
        assert abs(transformed[1] - point[1]) < 0.01, f"Y mismatch: {transformed[1]} != {point[1]}"
    
    print("  ✓ Identity transformation passed")


def test_transform_point_with_homography():
    """Test transform_point with a sample homography matrix."""
    print("\n=== Testing transform_point with sample homography ===")
    
    # Simple scaling and translation homography
    # Scale by 0.5 and translate by (100, 50)
    homography = np.array([
        [0.5, 0, 100],
        [0, 0.5, 50],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Test point (200, 300)
    # Expected: (200 * 0.5 + 100, 300 * 0.5 + 50) = (200, 200)
    point = (200, 300)
    transformed = transform_point(point, homography)
    print(f"  Input: {point} -> Output: {transformed}")
    print(f"  Expected: (200, 200)")
    
    assert abs(transformed[0] - 200) < 0.01, f"X mismatch: {transformed[0]} != 200"
    assert abs(transformed[1] - 200) < 0.01, f"Y mismatch: {transformed[1]} != 200"
    
    print("  ✓ Homography transformation passed")


def test_transform_points_batch():
    """Test transform_points with multiple points."""
    print("\n=== Testing transform_points with batch transformation ===")
    
    # Translation homography: shift by (50, 100)
    homography = np.array([
        [1, 0, 50],
        [0, 1, 100],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Test multiple points
    points = [(0, 0), (100, 200), (50, 50)]
    transformed = transform_points(points, homography)
    
    print(f"  Input points: {points}")
    print(f"  Transformed: {transformed}")
    
    # Expected results: each point shifted by (50, 100)
    expected = [(50, 100), (150, 300), (100, 150)]
    print(f"  Expected: {expected}")
    
    for i, (trans, exp) in enumerate(zip(transformed, expected)):
        assert abs(trans[0] - exp[0]) < 0.01, f"Point {i} X mismatch: {trans[0]} != {exp[0]}"
        assert abs(trans[1] - exp[1]) < 0.01, f"Point {i} Y mismatch: {trans[1]} != {exp[1]}"
    
    print("  ✓ Batch transformation passed")


def test_transform_points_empty():
    """Test transform_points with empty list."""
    print("\n=== Testing transform_points with empty list ===")
    
    identity_matrix = np.eye(3, dtype=np.float32)
    result = transform_points([], identity_matrix)
    
    print(f"  Input: [] -> Output: {result}")
    assert result == [], "Empty list should return empty list"
    
    print("  ✓ Empty list handling passed")


def test_validate_canvas_coordinates():
    """Test validate_canvas_coordinates with points inside and outside bounds."""
    print("\n=== Testing validate_canvas_coordinates ===")
    
    canvas_width = 800
    canvas_height = 600
    
    # Mix of valid and invalid points
    points = [
        (100, 200),      # Valid
        (400, 300),      # Valid
        (0, 0),          # Valid (boundary)
        (800, 600),      # Valid (boundary)
        (-10, 300),      # Invalid (x < 0)
        (900, 300),      # Invalid (x > width)
        (400, -50),      # Invalid (y < 0)
        (400, 700),      # Invalid (y > height)
        (1000, 1000),    # Invalid (both out of bounds)
    ]
    
    valid_points, invalid_count = validate_canvas_coordinates(points, canvas_width, canvas_height)
    
    print(f"  Canvas size: {canvas_width} x {canvas_height}")
    print(f"  Total points: {len(points)}")
    print(f"  Valid points: {len(valid_points)}")
    print(f"  Invalid count: {invalid_count}")
    print(f"  Valid points: {valid_points}")
    
    assert len(valid_points) == 4, f"Expected 4 valid points, got {len(valid_points)}"
    assert invalid_count == 5, f"Expected 5 invalid points, got {invalid_count}"
    
    # Check that valid points are correct
    expected_valid = [(100, 200), (400, 300), (0, 0), (800, 600)]
    for point in expected_valid:
        assert point in valid_points, f"Expected valid point {point} not in results"
    
    print("  ✓ Validation passed")


def test_coordinate_transformer():
    """Test CoordinateTransformer with mock calibration and detection data."""
    print("\n=== Testing CoordinateTransformer ===")
    
    # Create mock calibration data with 2 cameras
    calibration_data = {
        'canvas_width': 1000,
        'canvas_height': 800,
        'cameras': [
            {
                'camera_id': 'cam1',
                # Simple translation: shift by (100, 50)
                'homography_matrix': [
                    [1, 0, 100],
                    [0, 1, 50],
                    [0, 0, 1]
                ]
            },
            {
                'camera_id': 'cam2',
                # Simple translation: shift by (200, 100)
                'homography_matrix': [
                    [1, 0, 200],
                    [0, 1, 100],
                    [0, 0, 1]
                ]
            }
        ]
    }
    
    # Create transformer
    transformer = CoordinateTransformer(calibration_data)
    
    print(f"  Canvas dimensions: {transformer.canvas_width} x {transformer.canvas_height}")
    print(f"  Cameras calibrated: {list(transformer.camera_homographies.keys())}")
    
    # Create mock detections (foot positions) for each camera
    detections_per_camera = {
        'cam1': [
            {'foot_position': (50, 100), 'bbox': [0, 0, 100, 200]},
            {'foot_position': (150, 200), 'bbox': [50, 50, 150, 250]},
            {'foot_position': (250, 300), 'bbox': [100, 100, 200, 300]},
        ],
        'cam2': [
            {'foot_position': (100, 150), 'bbox': [0, 0, 100, 200]},
            {'foot_position': (200, 250), 'bbox': [50, 50, 150, 250]},
        ]
    }
    
    print("\n  Input detections:")
    for cam_id, detections in detections_per_camera.items():
        print(f"    {cam_id}: {[d['foot_position'] for d in detections]}")
    
    # Transform all detections
    canvas_positions = transformer.transform_detections(detections_per_camera)
    
    print(f"\n  Transformed canvas positions ({len(canvas_positions)} total):")
    for i, pos in enumerate(canvas_positions):
        print(f"    {i+1}. {pos}")
    
    # Expected results:
    # cam1 detections shifted by (100, 50):
    #   (50, 100) -> (150, 150)
    #   (150, 200) -> (250, 250)
    #   (250, 300) -> (350, 350)
    # cam2 detections shifted by (200, 100):
    #   (100, 150) -> (300, 250)
    #   (200, 250) -> (400, 350)
    expected_positions = [
        (150, 150), (250, 250), (350, 350),  # cam1
        (300, 250), (400, 350)                # cam2
    ]
    
    print(f"\n  Expected positions:")
    for i, pos in enumerate(expected_positions):
        print(f"    {i+1}. {pos}")
    
    # Verify we got the expected number of positions
    assert len(canvas_positions) == 5, f"Expected 5 positions, got {len(canvas_positions)}"
    
    # Verify each position matches expected (order should be preserved)
    for i, (actual, expected) in enumerate(zip(canvas_positions, expected_positions)):
        assert abs(actual[0] - expected[0]) < 0.01, f"Position {i} X mismatch: {actual[0]} != {expected[0]}"
        assert abs(actual[1] - expected[1]) < 0.01, f"Position {i} Y mismatch: {actual[1]} != {expected[1]}"
    
    # Verify all positions are within canvas bounds
    for pos in canvas_positions:
        assert 0 <= pos[0] <= 1000, f"X coordinate {pos[0]} out of bounds"
        assert 0 <= pos[1] <= 800, f"Y coordinate {pos[1]} out of bounds"
    
    print("  ✓ CoordinateTransformer passed")


def test_coordinate_transformer_out_of_bounds():
    """Test CoordinateTransformer with detections that transform outside canvas."""
    print("\n=== Testing CoordinateTransformer with out-of-bounds points ===")
    
    # Small canvas
    calibration_data = {
        'canvas_width': 500,
        'canvas_height': 400,
        'cameras': [
            {
                'camera_id': 'cam1',
                # Large translation that will push some points out of bounds
                'homography_matrix': [
                    [1, 0, 400],
                    [0, 1, 300],
                    [0, 0, 1]
                ]
            }
        ]
    }
    
    transformer = CoordinateTransformer(calibration_data)
    
    # Detections that will be both in and out of bounds after transformation
    detections_per_camera = {
        'cam1': [
            {'foot_position': (50, 50)},    # -> (450, 350) - valid
            {'foot_position': (150, 150)},  # -> (550, 450) - out of bounds
            {'foot_position': (10, 10)},    # -> (410, 310) - valid
        ]
    }
    
    print(f"  Canvas: {transformer.canvas_width} x {transformer.canvas_height}")
    print(f"  Input foot positions: {[d['foot_position'] for d in detections_per_camera['cam1']]}")
    
    canvas_positions = transformer.transform_detections(detections_per_camera)
    
    print(f"  Valid canvas positions: {canvas_positions}")
    print(f"  Number of valid positions: {len(canvas_positions)}")
    
    # Only 2 positions should be valid
    assert len(canvas_positions) == 2, f"Expected 2 valid positions, got {len(canvas_positions)}"
    
    # Verify valid positions are within bounds
    for pos in canvas_positions:
        assert 0 <= pos[0] <= 500, f"X coordinate {pos[0]} out of bounds"
        assert 0 <= pos[1] <= 400, f"Y coordinate {pos[1]} out of bounds"
    
    print("  ✓ Out-of-bounds filtering passed")


def test_coordinate_transformer_uncalibrated_camera():
    """Test CoordinateTransformer with detections from uncalibrated camera."""
    print("\n=== Testing CoordinateTransformer with uncalibrated camera ===")
    
    calibration_data = {
        'canvas_width': 1000,
        'canvas_height': 800,
        'cameras': [
            {
                'camera_id': 'cam1',
                'homography_matrix': [
                    [1, 0, 100],
                    [0, 1, 50],
                    [0, 0, 1]
                ]
            }
        ]
    }
    
    transformer = CoordinateTransformer(calibration_data)
    
    # Include detections from both calibrated and uncalibrated cameras
    detections_per_camera = {
        'cam1': [
            {'foot_position': (100, 100)},
        ],
        'cam_uncalibrated': [
            {'foot_position': (200, 200)},
            {'foot_position': (300, 300)},
        ]
    }
    
    print(f"  Calibrated cameras: {list(transformer.camera_homographies.keys())}")
    print(f"  Detection cameras: {list(detections_per_camera.keys())}")
    
    canvas_positions = transformer.transform_detections(detections_per_camera)
    
    print(f"  Canvas positions: {canvas_positions}")
    print(f"  Number of positions: {len(canvas_positions)}")
    
    # Only cam1 detections should be transformed (1 detection)
    assert len(canvas_positions) == 1, f"Expected 1 position from calibrated camera, got {len(canvas_positions)}"
    
    print("  ✓ Uncalibrated camera filtering passed")


def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("TRANSFORMER MODULE TEST SUITE")
    print("=" * 70)
    
    try:
        test_transform_point_identity()
        test_transform_point_with_homography()
        test_transform_points_batch()
        test_transform_points_empty()
        test_validate_canvas_coordinates()
        test_coordinate_transformer()
        test_coordinate_transformer_out_of_bounds()
        test_coordinate_transformer_uncalibrated_camera()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
