"""
Test script for person detection module.
Tests YOLOv8 detector on sample frames and visualizes results.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detector import PersonDetector


def create_synthetic_frame():
    """Create a synthetic test frame with a person-like rectangle."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a person-like shape (simulating a standing person)
    cv2.rectangle(frame, (250, 150), (390, 450), (100, 150, 200), -1)  # Body
    cv2.circle(frame, (320, 120), 30, (150, 180, 220), -1)  # Head
    return frame


def draw_detections(frame, detections):
    """Draw bounding boxes and foot positions on frame."""
    annotated = frame.copy()
    
    for det in detections:
        bbox = det['bbox']
        confidence = det['confidence']
        foot_pos = det['foot_position']
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence label
        label = f"Person {confidence:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw foot position (red circle)
        foot_x, foot_y = map(int, foot_pos)
        cv2.circle(annotated, (foot_x, foot_y), 5, (0, 0, 255), -1)
        
        # Draw line from bbox bottom to foot point
        cv2.line(annotated, ((x1 + x2) // 2, y2), (foot_x, foot_y), (255, 0, 0), 2)
    
    return annotated


def test_synthetic_frame():
    """Test detection on synthetic frame."""
    print("\n" + "="*60)
    print("TEST 1: Synthetic Frame Detection")
    print("="*60)
    
    # Create detector
    detector = PersonDetector(model_name='models/yolov8n.pt', confidence_threshold=0.3)
    
    # Create synthetic frame
    frame = create_synthetic_frame()
    print(f"Created synthetic frame: {frame.shape}")
    
    # Run detection
    detections = detector.detect_persons(frame)
    print(f"Detections found: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"\nDetection {i+1}:")
        print(f"  BBox: {det['bbox']}")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Foot Position: {det['foot_position']}")
    
    # Visualize
    annotated = draw_detections(frame, detections)
    cv2.imshow('Synthetic Frame Detection', annotated)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_real_video_frame():
    """Test detection on real video frame if available."""
    print("\n" + "="*60)
    print("TEST 2: Real Video Frame Detection")
    print("="*60)
    
    # Look for sample video in media directory
    media_dir = Path(__file__).parent.parent / 'media'
    video_files = list(media_dir.glob('*.mp4')) + list(media_dir.glob('*.avi'))
    
    if not video_files:
        print("No video files found in media/ directory. Skipping real video test.")
        return
    
    video_path = video_files[0]
    print(f"Using video: {video_path.name}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame from video")
        return
    
    print(f"Loaded frame: {frame.shape}")
    
    # Create detector
    detector = PersonDetector(model_name='models/yolov8n.pt', confidence_threshold=0.5)
    
    # Run detection
    detections = detector.detect_persons(frame)
    print(f"Detections found: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"\nDetection {i+1}:")
        print(f"  BBox: {[f'{x:.1f}' for x in det['bbox']]}")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Foot Position: ({det['foot_position'][0]:.1f}, {det['foot_position'][1]:.1f})")
    
    # Visualize
    annotated = draw_detections(frame, detections)
    cv2.imshow('Real Video Frame Detection', annotated)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_batch_processing():
    """Test batch detection on multiple frames."""
    print("\n" + "="*60)
    print("TEST 3: Batch Processing")
    print("="*60)
    
    detector = PersonDetector(model_name='models/yolov8n.pt', confidence_threshold=0.5)
    
    # Create multiple synthetic frames
    frames = [create_synthetic_frame() for _ in range(3)]
    print(f"Created {len(frames)} synthetic frames")
    
    # Test list-based batch processing
    detections_list = detector.detect_persons_batch(frames)
    print(f"\nList-based batch processing:")
    for i, dets in enumerate(detections_list):
        print(f"  Frame {i+1}: {len(dets)} detections")
    
    # Test dict-based batch processing (simulating multi-camera)
    frames_dict = {
        'cam1': frames[0],
        'cam2': frames[1],
        'cam3': frames[2]
    }
    detections_dict = detector.detect_persons_batch(frames_dict)
    print(f"\nDict-based batch processing:")
    for cam_id, dets in detections_dict.items():
        print(f"  {cam_id}: {len(dets)} detections")


def test_empty_frame():
    """Test detection on empty frame."""
    print("\n" + "="*60)
    print("TEST 4: Empty Frame Detection")
    print("="*60)
    
    detector = PersonDetector(model_name='models/yolov8n.pt', confidence_threshold=0.5)
    
    # Create empty frame (all black)
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detections = detector.detect_persons(empty_frame)
    print(f"Detections on empty frame: {len(detections)}")
    assert len(detections) == 0, "Empty frame should have no detections"
    print("✓ Empty frame test passed")


def main():
    """Run all detector tests."""
    print("\n" + "="*60)
    print("PERSON DETECTOR TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Synthetic frame
        test_synthetic_frame()
        
        # Test 2: Real video frame (if available)
        test_real_video_frame()
        
        # Test 3: Batch processing
        test_batch_processing()
        
        # Test 4: Empty frame
        test_empty_frame()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
