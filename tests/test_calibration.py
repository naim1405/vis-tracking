"""
Test script for calibration module.

Tests:
1. Point selection UI (manual interaction)
2. Homography computation with sample points
3. Homography transformation validation
4. Grid dimension calculation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from calibration import (
    PointSelector, 
    calculate_grid_dimensions, 
    compute_homography,
    calibrate_cameras,
    GridArrangementUI
)
from config import VIDEO_FILES, CANVAS_WIDTH, CANVAS_HEIGHT


def test_grid_dimensions():
    """Test grid layout calculation for different camera counts."""
    print("\n=== Test 1: Grid Dimensions ===")
    
    test_cases = [
        (1, (1, 1)),
        (2, (2, 1)),
        (3, (2, 2)),
        (4, (2, 2)),
        (5, (3, 2)),
        (6, (3, 2)),
        (7, (3, 3)),
        (8, (3, 3)),
        (9, (3, 3)),
    ]
    
    all_passed = True
    for num_cameras, expected in test_cases:
        result = calculate_grid_dimensions(num_cameras)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"{status} {num_cameras} cameras -> {result} (expected {expected})")
        if not passed:
            all_passed = False
    
    print(f"\n{'✓ PASSED' if all_passed else '✗ FAILED'}: Grid dimensions test")
    return all_passed


def test_homography_computation():
    """Test homography computation with known point correspondences."""
    print("\n=== Test 2: Homography Computation ===")
    
    # Test case 1: Simple rectangle to rectangle mapping
    print("\nTest case 1: Rectangle to rectangle")
    source_points = [
        (100, 100),   # top-left
        (500, 100),   # top-right
        (500, 400),   # bottom-right
        (100, 400)    # bottom-left
    ]
    
    destination_points = [
        (0, 0),
        (600, 0),
        (600, 600),
        (0, 600)
    ]
    
    try:
        H = compute_homography(source_points, destination_points)
        print(f"✓ Homography matrix computed successfully")
        print(f"  Matrix shape: {H.shape}")
        print(f"  Matrix determinant: {np.linalg.det(H):.6f}")
        print(f"  Matrix:\n{H}")
        
        # Test transformation on center point
        test_point = np.array([[[300, 250]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(test_point, H)
        tx, ty = transformed[0][0]
        print(f"  Transform test: (300, 250) -> ({tx:.1f}, {ty:.1f})")
        
        # Expected is roughly (300, 300) with some scaling
        expected_x = 300
        expected_y = 300
        tolerance = 50
        
        if abs(tx - expected_x) < tolerance and abs(ty - expected_y) < tolerance:
            print(f"  ✓ Transformation is reasonable")
            test1_passed = True
        else:
            print(f"  ✗ Transformation seems incorrect (expected ~{expected_x}, ~{expected_y})")
            test1_passed = False
        
    except Exception as e:
        print(f"✗ Failed to compute homography: {e}")
        test1_passed = False
    
    # Test case 2: Perspective transformation
    print("\nTest case 2: Perspective transformation")
    source_points = [
        (200, 150),   # top-left
        (800, 180),   # top-right
        (850, 700),   # bottom-right
        (150, 680)    # bottom-left
    ]
    
    destination_points = [
        (0, 0),
        (1200, 0),
        (1200, 1200),
        (0, 1200)
    ]
    
    try:
        H = compute_homography(source_points, destination_points)
        print(f"✓ Homography matrix computed successfully")
        print(f"  Matrix determinant: {np.linalg.det(H):.6f}")
        
        # Test if matrix is invertible
        H_inv = np.linalg.inv(H)
        print(f"  ✓ Matrix is invertible")
        test2_passed = True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        test2_passed = False
    
    # Test case 3: Invalid points (collinear)
    print("\nTest case 3: Invalid points (should fail)")
    source_points = [
        (100, 100),
        (200, 100),
        (300, 100),
        (400, 100)  # All on same line
    ]
    
    destination_points = [
        (0, 0),
        (300, 0),
        (300, 300),
        (0, 300)
    ]
    
    try:
        H = compute_homography(source_points, destination_points)
        det = np.linalg.det(H)
        if abs(det) < 1e-6:
            print(f"✓ Correctly detected singular matrix (det={det:.2e})")
            test3_passed = True
        else:
            print(f"✗ Should have failed with collinear points")
            test3_passed = False
    except RuntimeError as e:
        print(f"✓ Correctly raised error: {e}")
        test3_passed = True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        test3_passed = False
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\n{'✓ PASSED' if all_passed else '✗ FAILED'}: Homography computation test")
    return all_passed


def test_point_selector_manual():
    """
    Manual test for PointSelector UI.
    Requires user interaction.
    """
    print("\n=== Test 3: Point Selection (Manual) ===")
    print("This test requires user interaction.")
    
    # Check if test video exists
    if not VIDEO_FILES:
        print("✗ No video files configured in config.py")
        return False
    
    test_video = VIDEO_FILES[0]
    
    if not os.path.exists(test_video):
        print(f"✗ Test video not found: {test_video}")
        print("  Skipping manual point selection test")
        return None  # Skip, not a failure
    
    # Load first frame
    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {test_video}")
        cap.release()
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"✗ Cannot read frame from video")
        return False
    
    print(f"✓ Loaded test frame from: {test_video}")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run point selector
    try:
        selector = PointSelector(frame, 0)
        points = selector.select_points()
        
        print(f"✓ Point selection completed")
        print(f"  Selected points: {points}")
        
        # Validate we got 4 points
        if len(points) == 4:
            print(f"✓ Correct number of points")
            
            # Check points are within frame
            h, w = frame.shape[:2]
            all_valid = all(0 <= x < w and 0 <= y < h for x, y in points)
            
            if all_valid:
                print(f"✓ All points within frame bounds")
                return True
            else:
                print(f"✗ Some points outside frame bounds")
                return False
        else:
            print(f"✗ Wrong number of points: {len(points)}")
            return False
            
    except RuntimeError as e:
        print(f"  User cancelled: {e}")
        return None  # User cancelled, not a test failure
    except Exception as e:
        print(f"✗ Error during point selection: {e}")
        return False


def test_homography_transformation():
    """Test actual point transformation using computed homography."""
    print("\n=== Test 4: Homography Transformation ===")
    
    # Create test homography
    source_points = [
        (0, 0),
        (640, 0),
        (640, 480),
        (0, 480)
    ]
    
    destination_points = [
        (100, 100),
        (700, 100),
        (700, 700),
        (100, 700)
    ]
    
    try:
        H = compute_homography(source_points, destination_points)
        
        # Test transforming multiple points
        test_points = [
            (0, 0),      # Corner
            (640, 480),  # Opposite corner
            (320, 240),  # Center
            (160, 120),  # Random point
        ]
        
        print("Transforming test points:")
        all_valid = True
        
        for src_pt in test_points:
            pt_array = np.array([[src_pt]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(pt_array, H)
            dx, dy = dst_pt[0][0]
            print(f"  {src_pt} -> ({dx:.1f}, {dy:.1f})")
            
            # Check transformed point is within reasonable bounds
            if not (0 <= dx <= 1000 and 0 <= dy <= 1000):
                print(f"    ✗ Transformed point outside expected range")
                all_valid = False
        
        if all_valid:
            print(f"✓ All transformations within expected range")
            return True
        else:
            print(f"✗ Some transformations out of range")
            return False
            
    except Exception as e:
        print(f"✗ Error during transformation: {e}")
        return False


def test_full_calibration():
    """
    Test full calibration workflow with all configured cameras.
    This is a manual test requiring user interaction.
    """
    print("\n=== Test 5: Full Calibration Workflow (Manual) ===")
    
    # Check videos exist
    missing_videos = []
    for video_file in VIDEO_FILES:
        if not os.path.exists(video_file):
            missing_videos.append(video_file)
    
    if missing_videos:
        print(f"✗ Missing video files:")
        for video in missing_videos:
            print(f"  - {video}")
        print("  Skipping full calibration test")
        return None
    
    print(f"Testing with {len(VIDEO_FILES)} cameras")
    print("This will require you to:")
    print("  1. Select points for each camera")
    print("  2. Confirm or customize grid arrangement")
    
    response = input("\nProceed with full calibration test? (y/n): ")
    if response.lower() != 'y':
        print("  Test skipped by user")
        return None
    
    try:
        calibration_data = calibrate_cameras(
            VIDEO_FILES,
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        )
        
        print("\n✓ Calibration completed successfully")
        print(f"  Canvas: {calibration_data['canvas_width']}x{calibration_data['canvas_height']}")
        print(f"  Grid: {calibration_data['grid_layout']}")
        print(f"  Cameras calibrated: {len(calibration_data['cameras'])}")
        
        # Validate calibration data structure
        all_valid = True
        for camera in calibration_data['cameras']:
            camera_id = camera['camera_id']
            
            # Check required fields
            required_fields = [
                'camera_id', 'video_file', 'source_points',
                'destination_points', 'homography_matrix', 'grid_cell'
            ]
            
            for field in required_fields:
                if field not in camera:
                    print(f"  ✗ Camera {camera_id} missing field: {field}")
                    all_valid = False
            
            # Check point counts
            if len(camera['source_points']) != 4:
                print(f"  ✗ Camera {camera_id} has {len(camera['source_points'])} source points (expected 4)")
                all_valid = False
            
            if len(camera['destination_points']) != 4:
                print(f"  ✗ Camera {camera_id} has {len(camera['destination_points'])} dest points (expected 4)")
                all_valid = False
            
            # Check homography matrix shape
            H = camera['homography_matrix']
            if H.shape != (3, 3):
                print(f"  ✗ Camera {camera_id} homography has wrong shape: {H.shape}")
                all_valid = False
        
        if all_valid:
            print(f"✓ All calibration data validated")
            
            # Print summary
            print("\nCalibration Summary:")
            for camera in calibration_data['cameras']:
                print(f"  Camera {camera['camera_id']}:")
                print(f"    Grid cell: {camera['grid_cell']}")
                print(f"    Source points: {camera['source_points']}")
                print(f"    Dest rect: {camera['destination_points'][0]} to {camera['destination_points'][2]}")
            
            return True
        else:
            print(f"✗ Calibration data validation failed")
            return False
            
    except Exception as e:
        print(f"✗ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_arrangement_ui():
    """
    Test GridArrangementUI with mock camera frames.
    This is a manual test requiring user interaction.
    """
    print("\n=== Test 6: Grid Arrangement UI (Manual) ===")
    
    # Check if test video exists
    if not VIDEO_FILES or not os.path.exists(VIDEO_FILES[0]):
        print("✗ No test video available")
        print("  Skipping grid arrangement UI test")
        return None
    
    print("This test will display the grid arrangement UI.")
    response = input("Proceed with grid arrangement test? (y/n): ")
    if response.lower() != 'y':
        print("  Test skipped by user")
        return None
    
    try:
        # Load test frames
        num_test_cameras = min(3, len(VIDEO_FILES))  # Use up to 3 cameras
        camera_frames = {}
        
        for i in range(num_test_cameras):
            if i < len(VIDEO_FILES) and os.path.exists(VIDEO_FILES[i]):
                cap = cv2.VideoCapture(VIDEO_FILES[i])
                ret, frame = cap.read()
                cap.release()
                if ret:
                    camera_frames[i] = frame
            else:
                # Create dummy frame if video not available
                camera_frames[i] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print(f"✓ Loaded {len(camera_frames)} camera frames")
        
        # Create GridArrangementUI
        arrangement_ui = GridArrangementUI(
            num_test_cameras,
            camera_frames,
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        )
        
        print("Testing grid arrangement UI...")
        print("Instructions:")
        print("  - Press 'c' to confirm default arrangement")
        print("  - Press 'e' to edit (then enter custom positions)")
        print("  - Press 'r' to reset to default")
        
        # Run arrangement UI
        camera_positions = arrangement_ui.arrange_cameras_simple()
        
        print("\n✓ Grid arrangement completed")
        print("Final camera positions:")
        for camera_id in sorted(camera_positions.keys()):
            row, col = camera_positions[camera_id]
            print(f"  Camera {camera_id} -> Grid cell ({row}, {col})")
        
        # Validate results
        if len(camera_positions) != num_test_cameras:
            print(f"✗ Wrong number of camera positions: {len(camera_positions)} (expected {num_test_cameras})")
            return False
        
        # Check no duplicate positions
        position_set = set(camera_positions.values())
        if len(position_set) != len(camera_positions):
            print(f"✗ Duplicate grid positions detected")
            return False
        
        print("✓ All positions valid (no duplicates)")
        return True
        
    except Exception as e:
        print(f"✗ Grid arrangement UI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arrangement_input_parsing():
    """Test input parsing for grid arrangement."""
    print("\n=== Test 7: Arrangement Input Parsing ===")
    
    # Create dummy frames
    camera_frames = {
        0: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        1: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        2: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    
    arrangement_ui = GridArrangementUI(3, camera_frames, 1200, 1200)
    
    # Test valid inputs
    test_cases = [
        ("0:0,0 1:0,1 2:1,0", True, "Valid arrangement"),
        ("0:0,1 1:0,0 2:1,1", True, "Valid arrangement (swapped)"),
        ("0:0,0 1:0,0 2:1,0", False, "Duplicate positions"),
        ("0:0,0 1:0,1", False, "Missing camera"),
        ("0:0,0 1:0,1 2:5,0", False, "Invalid row"),
        ("0:0,0 1:0,5 2:1,0", False, "Invalid col"),
        ("invalid", False, "Invalid format"),
    ]
    
    all_passed = True
    
    for input_str, should_succeed, description in test_cases:
        result = arrangement_ui._parse_arrangement_input(input_str)
        success = result is not None
        
        if success == should_succeed:
            status = "✓"
        else:
            status = "✗"
            all_passed = False
        
        expected = "should succeed" if should_succeed else "should fail"
        print(f"{status} {description}: {expected}")
        if result is not None:
            print(f"    Parsed: {result}")
    
    print(f"\n{'✓ PASSED' if all_passed else '✗ FAILED'}: Arrangement input parsing test")
    return all_passed


def main():
    """Run all calibration tests."""
    print("=" * 60)
    print("CALIBRATION MODULE TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Automatic tests
    results['grid_dimensions'] = test_grid_dimensions()
    results['homography_computation'] = test_homography_computation()
    results['homography_transformation'] = test_homography_transformation()
    results['arrangement_input_parsing'] = test_arrangement_input_parsing()
    
    # Manual tests (optional)
    print("\n" + "=" * 60)
    print("MANUAL TESTS (Require User Interaction)")
    print("=" * 60)
    
    response = input("\nRun manual point selection test? (y/n): ")
    if response.lower() == 'y':
        results['point_selector'] = test_point_selector_manual()
    else:
        results['point_selector'] = None
        print("  Test skipped")
    
    response = input("\nRun grid arrangement UI test? (y/n): ")
    if response.lower() == 'y':
        results['grid_arrangement_ui'] = test_grid_arrangement_ui()
    else:
        results['grid_arrangement_ui'] = None
        print("  Test skipped")
    
    results['full_calibration'] = test_full_calibration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for v in results.values() if v is True)
    failed_count = sum(1 for v in results.values() if v is False)
    skipped_count = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊗ SKIPPED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count} passed, {failed_count} failed, {skipped_count} skipped")
    
    if failed_count == 0:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        return 0
    else:
        print(f"\n✗✗✗ {failed_count} TEST(S) FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
