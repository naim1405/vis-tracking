"""
Phase 6 test script for multi-camera person tracking system.

This script validates the integration of all modules in main.py,
including video loading, frame reading, and error handling.
"""

import sys
import os
import cv2
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import load_videos, read_synchronized_frames
from config import VIDEO_FILES

print("=" * 60)
print("PHASE 6 TEST - Integration Testing")
print("=" * 60)
print()


def create_test_video(filepath: str, num_frames: int = 30, width: int = 640, height: int = 480, fps: float = 30.0):
    """
    Create a test video file with moving circle.
    
    Args:
        filepath: Path to save the video
        num_frames: Number of frames to generate
        width: Video width
        height: Video height
        fps: Frames per second
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create frame with moving circle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Gray background
        
        # Draw moving circle
        x = int((i / num_frames) * width)
        y = height // 2
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {i+1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {filepath} ({num_frames} frames, {width}x{height}, {fps} fps)")


print("1. Setting up test environment...")
print("-" * 60)

# Create videos directory if it doesn't exist
videos_dir = "/home/ezio/Documents/work/vis-tracking/videos"
os.makedirs(videos_dir, exist_ok=True)
print(f"✓ Videos directory created/verified: {videos_dir}")

# Create test videos
test_video_paths = [
    os.path.join(videos_dir, "test_camera1.mp4"),
    os.path.join(videos_dir, "test_camera2.mp4"),
]

print("\n2. Creating test videos...")
print("-" * 60)

create_test_video(test_video_paths[0], num_frames=30, width=640, height=480, fps=30.0)
create_test_video(test_video_paths[1], num_frames=25, width=800, height=600, fps=25.0)

print("\n3. Testing load_videos function...")
print("-" * 60)

try:
    captures, metadata = load_videos(test_video_paths)
    
    # Verify captures
    assert len(captures) == 2, f"Expected 2 captures, got {len(captures)}"
    assert 0 in captures, "Camera 0 not in captures"
    assert 1 in captures, "Camera 1 not in captures"
    assert all(isinstance(cap, cv2.VideoCapture) for cap in captures.values()), "Invalid capture objects"
    print("✓ Captures created successfully")
    
    # Verify metadata structure
    assert 'fps' in metadata, "Missing 'fps' in metadata"
    assert 'frame_counts' in metadata, "Missing 'frame_counts' in metadata"
    assert 'resolutions' in metadata, "Missing 'resolutions' in metadata"
    print("✓ Metadata structure correct")
    
    # Verify metadata values
    assert len(metadata['fps']) == 2, "Wrong number of fps values"
    assert len(metadata['frame_counts']) == 2, "Wrong number of frame_count values"
    assert len(metadata['resolutions']) == 2, "Wrong number of resolution values"
    print("✓ Metadata lists have correct length")
    
    # Check specific values
    assert metadata['fps'][0] == 30.0, f"Camera 0 FPS wrong: {metadata['fps'][0]}"
    assert metadata['fps'][1] == 25.0, f"Camera 1 FPS wrong: {metadata['fps'][1]}"
    assert metadata['frame_counts'][0] == 30, f"Camera 0 frame count wrong: {metadata['frame_counts'][0]}"
    assert metadata['frame_counts'][1] == 25, f"Camera 1 frame count wrong: {metadata['frame_counts'][1]}"
    assert metadata['resolutions'][0] == (640, 480), f"Camera 0 resolution wrong: {metadata['resolutions'][0]}"
    assert metadata['resolutions'][1] == (800, 600), f"Camera 1 resolution wrong: {metadata['resolutions'][1]}"
    print("✓ Metadata values correct")
    
    print("✓ load_videos test passed")
    
except Exception as e:
    print(f"✗ load_videos test failed: {e}")
    sys.exit(1)

print("\n4. Testing read_synchronized_frames function...")
print("-" * 60)

try:
    # Reset to beginning
    for cap in captures.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Read first frame
    frames_dict, all_finished = read_synchronized_frames(captures)
    
    assert not all_finished, "Should not be finished on first frame"
    assert len(frames_dict) == 2, f"Expected 2 frames, got {len(frames_dict)}"
    assert 0 in frames_dict, "Camera 0 frame missing"
    assert 1 in frames_dict, "Camera 1 frame missing"
    assert frames_dict[0].shape == (480, 640, 3), f"Camera 0 frame shape wrong: {frames_dict[0].shape}"
    assert frames_dict[1].shape == (600, 800, 3), f"Camera 1 frame shape wrong: {frames_dict[1].shape}"
    print("✓ First frame read successfully")
    
    # Read all frames
    frame_count = 1
    frames_with_both = 0
    frames_with_one = 0
    
    while True:
        frames_dict, all_finished = read_synchronized_frames(captures)
        if all_finished:
            break
        
        if len(frames_dict) == 2:
            frames_with_both += 1
        elif len(frames_dict) == 1:
            frames_with_one += 1
        
        frame_count += 1
    
    # Camera 0 has 30 frames, Camera 1 has 25 frames
    # We should read 30 total frames: 25 with both cameras, 5 with only camera 0
    assert frame_count == 30, f"Expected to read 30 frames, got {frame_count}"
    assert frames_with_both == 24, f"Expected 24 frames with both cameras, got {frames_with_both}"
    assert frames_with_one == 5, f"Expected 5 frames with one camera, got {frames_with_one}"
    print(f"✓ Read all frames correctly ({frames_with_both} with both cameras, {frames_with_one} with one)")
    
    # Verify finished state
    frames_dict, all_finished = read_synchronized_frames(captures)
    assert all_finished, "Should be finished after all frames read"
    assert len(frames_dict) == 0, f"Should have no frames when finished, got {len(frames_dict)}"
    print("✓ Finished state correct")
    
    print("✓ read_synchronized_frames test passed")
    
except Exception as e:
    print(f"✗ read_synchronized_frames test failed: {e}")
    sys.exit(1)

print("\n5. Testing error handling - invalid video path...")
print("-" * 60)

try:
    invalid_paths = ["/nonexistent/video.mp4"]
    try:
        load_videos(invalid_paths)
        print("✗ Should have raised ValueError for invalid path")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
except Exception as e:
    print(f"✗ Error handling test failed: {e}")
    sys.exit(1)

print("\n6. Testing partial frame reads (camera desync)...")
print("-" * 60)

try:
    # Reset cameras
    for cap in captures.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Manually advance camera 0 ahead
    for i in range(10):
        captures[0].read()
    
    # Now read from both - should get frames from both
    frames_dict, all_finished = read_synchronized_frames(captures)
    
    assert not all_finished, "Should not be finished"
    assert 0 in frames_dict, "Should still read from camera 0"
    assert 1 in frames_dict, "Should still read from camera 1"
    print("✓ Handles desynchronized cameras correctly")
    
except Exception as e:
    print(f"✗ Desync test failed: {e}")
    sys.exit(1)

print("\n7. Cleanup...")
print("-" * 60)

# Release captures
for cap in captures.values():
    cap.release()
print("✓ Video captures released")

# Clean up test videos
for video_path in test_video_paths:
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"✓ Removed test video: {video_path}")

print()
print("=" * 60)
print("PHASE 6 TEST COMPLETED SUCCESSFULLY")
print("=" * 60)
print()
print("Integration functions validated:")
print("  ✓ load_videos() - loads videos and extracts metadata")
print("  ✓ read_synchronized_frames() - reads frames from all cameras")
print("  ✓ Error handling - invalid paths raise ValueError")
print("  ✓ Desynchronization handling - continues with available frames")
print()
print("Next step: Run full system with actual videos")
print("  python main.py")
