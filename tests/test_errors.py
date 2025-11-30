"""
Test error handling and validation for the multi-camera person tracking system.

This module tests:
- Invalid video paths
- Invalid calibration points (wrong count, collinear)
- Invalid homography matrices
- Empty detections
- Error recovery scenarios
"""

import os
import sys
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    validate_video_files,
    validate_point_count,
    validate_points_not_collinear,
    validate_homography_matrix
)
from calibration import compute_homography
from detector import PersonDetector

try:
    import pytest
    PYTEST_AVAILABLE = True
    
    # Define pytest decorators
    skipif = pytest.mark.skipif
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None
    
    # Mock pytest decorators for when pytest is not available
    class MockMark:
        def skipif(self, *args, **kwargs):
            return lambda func: func
    
    class MockPytest:
        mark = MockMark()
        
        @staticmethod
        def raises(*args, **kwargs):
            class RaisesContext:
                def __init__(self, exc_type, match=None):
                    self.exc_type = exc_type
                    self.match = match
                
                def __enter__(self):
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {self.exc_type} but no exception was raised")
                    if not issubclass(exc_type, self.exc_type):
                        return False  # Re-raise the exception
                    if self.match and self.match not in str(exc_val):
                        raise AssertionError(f"Exception message '{exc_val}' does not match '{self.match}'")
                    return True  # Suppress the exception
            
            return RaisesContext(*args, **kwargs)
    
    pytest = MockPytest()


class TestVideoValidation:
    """Test video file validation."""
    
    def test_no_video_files(self):
        """Empty video list should raise ValueError."""
        with pytest.raises(ValueError, match="No video files specified"):
            validate_video_files([])
    
    def test_nonexistent_video_file(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            validate_video_files(["/nonexistent/path/video.mp4"])
    
    def test_invalid_video_format(self, tmp_path):
        """Invalid video format should raise ValueError."""
        # Create a text file pretending to be a video
        fake_video = tmp_path / "fake_video.mp4"
        fake_video.write_text("This is not a video file")
        
        with pytest.raises(ValueError, match="Cannot open video file"):
            validate_video_files([str(fake_video)])
    
    def test_valid_video_files(self):
        """Valid video files should pass validation."""
        # Note: This test requires actual video files
        # In a real scenario, you would use test fixtures
        pass


class TestPointValidation:
    """Test calibration point validation."""
    
    def test_validate_point_count_correct(self):
        """Correct number of points should not raise."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        validate_point_count(points, 4)  # Should not raise
    
    def test_validate_point_count_too_few(self):
        """Too few points should raise ValueError."""
        points = [(0, 0), (100, 0), (100, 100)]
        with pytest.raises(ValueError, match="Expected 4 points, got 3"):
            validate_point_count(points, 4)
    
    def test_validate_point_count_too_many(self):
        """Too many points should raise ValueError."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100), (50, 50)]
        with pytest.raises(ValueError, match="Expected 4 points, got 5"):
            validate_point_count(points, 4)
    
    def test_validate_points_not_collinear_valid(self):
        """Valid quadrilateral should not raise."""
        # Large square
        points = [(0, 0), (200, 0), (200, 200), (0, 200)]
        validate_points_not_collinear(points)  # Should not raise
    
    def test_validate_points_collinear(self):
        """Collinear points should raise ValueError."""
        # All points on a line
        points = [(0, 0), (50, 0), (100, 0), (150, 0)]
        with pytest.raises(ValueError, match="too close or collinear"):
            validate_points_not_collinear(points)
    
    def test_validate_points_too_close(self):
        """Points too close together should raise ValueError."""
        # Very small area (< 100 pixels²)
        points = [(0, 0), (5, 0), (5, 5), (0, 5)]
        with pytest.raises(ValueError, match="too close or collinear"):
            validate_points_not_collinear(points)
    
    def test_validate_points_barely_valid(self):
        """Points with area just above threshold should pass."""
        # Area = 101 pixels² (just above threshold)
        points = [(0, 0), (10.1, 0), (10.1, 10), (0, 10)]
        validate_points_not_collinear(points)  # Should not raise


class TestHomographyValidation:
    """Test homography matrix validation."""
    
    def test_validate_homography_none(self):
        """None matrix should raise ValueError."""
        with pytest.raises(ValueError, match="Homography computation failed"):
            validate_homography_matrix(None)
    
    def test_validate_homography_singular(self):
        """Singular matrix should raise ValueError."""
        # Create a singular matrix (determinant = 0)
        singular_matrix = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9]
        ], dtype=np.float32)
        
        with pytest.raises(ValueError, match="singular"):
            validate_homography_matrix(singular_matrix)
    
    def test_validate_homography_valid(self):
        """Valid homography matrix should not raise."""
        # Identity-like matrix with non-zero determinant
        valid_matrix = np.eye(3, dtype=np.float32)
        validate_homography_matrix(valid_matrix)  # Should not raise


class TestHomographyComputation:
    """Test homography computation with error handling."""
    
    def test_compute_homography_valid(self):
        """Valid points should compute valid homography."""
        source = [(0, 0), (100, 0), (100, 100), (0, 100)]
        destination = [(0, 0), (200, 0), (200, 200), (0, 200)]
        
        H = compute_homography(source, destination)
        assert H is not None
        assert H.shape == (3, 3)
        assert abs(np.linalg.det(H)) > 1e-6
    
    def test_compute_homography_collinear_source(self):
        """Collinear source points should raise RuntimeError."""
        source = [(0, 0), (50, 0), (100, 0), (150, 0)]  # Collinear
        destination = [(0, 0), (200, 0), (200, 200), (0, 200)]
        
        with pytest.raises(RuntimeError):
            compute_homography(source, destination)
    
    def test_compute_homography_collinear_destination(self):
        """Collinear destination points should raise RuntimeError."""
        source = [(0, 0), (100, 0), (100, 100), (0, 100)]
        destination = [(0, 0), (50, 0), (100, 0), (150, 0)]  # Collinear
        
        with pytest.raises(RuntimeError):
            compute_homography(source, destination)


class TestDetectorErrorHandling:
    """Test detector error handling."""
    
    def test_detector_invalid_model(self):
        """Invalid model path should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
            PersonDetector(model_name="nonexistent_model.pt")
    
    def test_detector_none_frame(self):
        """None frame should return empty list, not crash."""
        detector = PersonDetector()
        detections = detector.detect_persons(None)
        assert detections == []
    
    def test_detector_empty_frame(self):
        """Empty frame should return empty list, not crash."""
        detector = PersonDetector()
        empty_frame = np.array([], dtype=np.uint8)
        detections = detector.detect_persons(empty_frame)
        assert detections == []
    
    def test_detector_black_frame(self):
        """Black frame with no people should return empty list."""
        detector = PersonDetector()
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_persons(black_frame)
        assert isinstance(detections, list)
        # May be empty or have false positives, but shouldn't crash


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_points_form_triangle(self):
        """Three distinct points should fail validation."""
        points = [(0, 0), (100, 0), (50, 100)]
        with pytest.raises(ValueError, match="Expected 4 points"):
            validate_point_count(points, 4)
    
    def test_duplicate_points(self):
        """Duplicate points should fail collinear check."""
        points = [(0, 0), (0, 0), (100, 100), (100, 100)]
        with pytest.raises(ValueError, match="too close or collinear"):
            validate_points_not_collinear(points)
    
    def test_negative_coordinates(self):
        """Negative coordinates should be valid if forming proper quadrilateral."""
        points = [(-100, -100), (100, -100), (100, 100), (-100, 100)]
        validate_point_count(points, 4)
        validate_points_not_collinear(points)  # Should not raise


def run_error_tests():
    """Run all error handling tests manually (without pytest runner)."""
    print("=" * 60)
    print("PHASE 7 - Error Handling Tests")
    print("=" * 60)
    
    test_count = 0
    passed = 0
    failed = 0
    
    # Test 1: No video files
    test_count += 1
    try:
        validate_video_files([])
        print(f"✗ Test {test_count}: No video files - should have raised ValueError")
        failed += 1
    except ValueError as e:
        if "No video files specified" in str(e):
            print(f"✓ Test {test_count}: No video files - correct error")
            passed += 1
        else:
            print(f"✗ Test {test_count}: Wrong error message: {e}")
            failed += 1
    
    # Test 2: Nonexistent video file
    test_count += 1
    try:
        validate_video_files(["/nonexistent/video.mp4"])
        print(f"✗ Test {test_count}: Nonexistent file - should have raised FileNotFoundError")
        failed += 1
    except FileNotFoundError as e:
        print(f"✓ Test {test_count}: Nonexistent file - correct error")
        passed += 1
    
    # Test 3: Wrong point count
    test_count += 1
    try:
        validate_point_count([(0, 0), (100, 0), (100, 100)], 4)
        print(f"✗ Test {test_count}: Wrong point count - should have raised ValueError")
        failed += 1
    except ValueError as e:
        if "Expected 4 points, got 3" in str(e):
            print(f"✓ Test {test_count}: Wrong point count - correct error")
            passed += 1
        else:
            print(f"✗ Test {test_count}: Wrong error message: {e}")
            failed += 1
    
    # Test 4: Collinear points
    test_count += 1
    try:
        validate_points_not_collinear([(0, 0), (50, 0), (100, 0), (150, 0)])
        print(f"✗ Test {test_count}: Collinear points - should have raised ValueError")
        failed += 1
    except ValueError as e:
        if "too close or collinear" in str(e):
            print(f"✓ Test {test_count}: Collinear points - correct error")
            passed += 1
        else:
            print(f"✗ Test {test_count}: Wrong error message: {e}")
            failed += 1
    
    # Test 5: Valid points
    test_count += 1
    try:
        validate_point_count([(0, 0), (100, 0), (100, 100), (0, 100)], 4)
        validate_points_not_collinear([(0, 0), (200, 0), (200, 200), (0, 200)])
        print(f"✓ Test {test_count}: Valid points - no error raised")
        passed += 1
    except Exception as e:
        print(f"✗ Test {test_count}: Valid points raised error: {e}")
        failed += 1
    
    # Test 6: Singular matrix
    test_count += 1
    try:
        singular = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=np.float32)
        validate_homography_matrix(singular)
        print(f"✗ Test {test_count}: Singular matrix - should have raised ValueError")
        failed += 1
    except ValueError as e:
        if "singular" in str(e):
            print(f"✓ Test {test_count}: Singular matrix - correct error")
            passed += 1
        else:
            print(f"✗ Test {test_count}: Wrong error message: {e}")
            failed += 1
    
    # Test 7: Valid homography matrix
    test_count += 1
    try:
        validate_homography_matrix(np.eye(3, dtype=np.float32))
        print(f"✓ Test {test_count}: Valid matrix - no error raised")
        passed += 1
    except Exception as e:
        print(f"✗ Test {test_count}: Valid matrix raised error: {e}")
        failed += 1
    
    # Test 8: None frame to detector
    test_count += 1
    try:
        detector = PersonDetector()
        detections = detector.detect_persons(None)
        if detections == []:
            print(f"✓ Test {test_count}: None frame - gracefully returned empty list")
            passed += 1
        else:
            print(f"✗ Test {test_count}: None frame - unexpected result: {detections}")
            failed += 1
    except Exception as e:
        print(f"✗ Test {test_count}: None frame - raised error: {e}")
        failed += 1
    
    # Test 9: Empty frame to detector
    test_count += 1
    try:
        detector = PersonDetector()
        empty = np.array([], dtype=np.uint8)
        detections = detector.detect_persons(empty)
        if detections == []:
            print(f"✓ Test {test_count}: Empty frame - gracefully returned empty list")
            passed += 1
        else:
            print(f"✗ Test {test_count}: Empty frame - unexpected result: {detections}")
            failed += 1
    except Exception as e:
        print(f"✗ Test {test_count}: Empty frame - raised error: {e}")
        failed += 1
    
    # Test 10: Invalid model path
    test_count += 1
    try:
        PersonDetector(model_name="nonexistent.pt")
        print(f"✗ Test {test_count}: Invalid model - should have raised RuntimeError")
        failed += 1
    except RuntimeError as e:
        if "Failed to load YOLO model" in str(e):
            print(f"✓ Test {test_count}: Invalid model - correct error")
            passed += 1
        else:
            print(f"✗ Test {test_count}: Wrong error message: {e}")
            failed += 1
    
    # Summary
    print("=" * 60)
    print(f"Test Results: {passed}/{test_count} passed, {failed}/{test_count} failed")
    if failed == 0:
        print("✓ All error handling tests passed!")
    print("=" * 60)
    
    return passed, failed


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    if PYTEST_AVAILABLE:
        print("Running tests with pytest...")
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, running manual tests...")
        run_error_tests()
