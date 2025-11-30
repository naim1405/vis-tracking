"""
Phase 1 test script for multi-camera person tracking system.

This script validates the basic project structure, configuration,
and utility functions created in Phase 1.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all modules to verify no syntax errors
print("=" * 60)
print("PHASE 1 TEST - Multi-Camera Person Tracking System")
print("=" * 60)
print()

print("1. Testing module imports...")
print("-" * 60)

try:
    import config
    print("✓ config.py imported successfully")
except Exception as e:
    print(f"✗ Failed to import config.py: {e}")
    sys.exit(1)

try:
    import utils
    print("✓ utils.py imported successfully")
except Exception as e:
    print(f"✗ Failed to import utils.py: {e}")
    sys.exit(1)

try:
    import calibration
    print("✓ calibration.py imported successfully")
except Exception as e:
    print(f"✗ Failed to import calibration.py: {e}")
    sys.exit(1)

try:
    import detector
    print("✓ detector.py imported successfully")
except Exception as e:
    print(f"✗ Failed to import detector.py: {e}")
    sys.exit(1)

try:
    import transformer
    print("✓ transformer.py imported successfully")
except Exception as e:
    print(f"✗ Failed to import transformer.py: {e}")
    sys.exit(1)

try:
    import visualizer
    print("✓ visualizer.py imported successfully")
except Exception as e:
    print(f"✗ Failed to import visualizer.py: {e}")
    sys.exit(1)

try:
    import main
    print("✓ main.py imported successfully")
except Exception as e:
    print(f"✗ Failed to import main.py: {e}")
    sys.exit(1)

print()
print("2. Printing configuration values...")
print("-" * 60)

print(f"VIDEO_FILES: {config.VIDEO_FILES}")
print(f"DETECTION_CONFIDENCE_THRESHOLD: {config.DETECTION_CONFIDENCE_THRESHOLD}")
print(f"YOLO_MODEL: {config.YOLO_MODEL}")
print(f"CANVAS_WIDTH: {config.CANVAS_WIDTH}")
print(f"CANVAS_HEIGHT: {config.CANVAS_HEIGHT}")
print(f"CANVAS_BACKGROUND_COLOR: {config.CANVAS_BACKGROUND_COLOR}")
print(f"DOT_COLOR: {config.DOT_COLOR}")
print(f"DOT_RADIUS: {config.DOT_RADIUS}")
print(f"DOT_THICKNESS: {config.DOT_THICKNESS}")
print(f"POINT_COLOR: {config.POINT_COLOR}")
print(f"POINT_RADIUS: {config.POINT_RADIUS}")

print()
print("3. Testing calculate_grid_dimensions function...")
print("-" * 60)

test_cases = [1, 2, 4, 6, 9]

for num_cams in test_cases:
    rows, cols = utils.calculate_grid_dimensions(num_cams)
    print(f"  {num_cams} camera(s) -> {rows} row(s) × {cols} col(s) (capacity: {rows * cols})")

print()
print("4. Testing validate_calibration_points function...")
print("-" * 60)

# Test valid quadrilateral
valid_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
is_valid, msg = utils.validate_calibration_points(valid_points)
print(f"  Valid square: {is_valid} {f'({msg})' if msg else ''}")

# Test invalid cases
invalid_cases = [
    ([(0, 0), (50, 50), (100, 100)], "3 points (need 4)"),
    ([(0, 0), (1, 1), (2, 2), (3, 3)], "collinear points"),
    ([(0, 0), (100, 0), (100, 100), ("invalid", 100)], "non-numeric value"),
]

for points, description in invalid_cases:
    is_valid, msg = utils.validate_calibration_points(points)
    print(f"  {description}: {is_valid} ({msg})")

print()
print("=" * 60)
print("PHASE 1 TEST COMPLETED SUCCESSFULLY")
print("=" * 60)
