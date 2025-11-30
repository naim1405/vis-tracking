"""
Main entry point for multi-camera person tracking system.

This module coordinates the entire tracking workflow:
1. Load configuration and initialize components
2. Run calibration mode to set up camera-to-floor transformations
3. Run detection and tracking mode to process video feeds
4. Display results on 2D floor map in real-time
"""

import cv2
import sys
import traceback
from typing import Dict, Tuple, List

from config import VIDEO_FILES, CANVAS_WIDTH, CANVAS_HEIGHT
from utils import calculate_grid_dimensions, validate_video_files
from calibration import calibrate_cameras
from detector import PersonDetector
from transformer import CoordinateTransformer
from visualizer import MapVisualizer


def load_videos(video_files: List[str]) -> Tuple[Dict[int, cv2.VideoCapture], Dict[str, List]]:
    """
    Load video files and extract metadata.
    
    Args:
        video_files: List of video file paths
        
    Returns:
        Tuple of (captures, metadata) where:
        - captures: dict mapping camera_id -> VideoCapture
        - metadata: dict with 'fps', 'frame_counts', 'resolutions' lists
        
    Raises:
        ValueError: If any video cannot be opened
    """
    captures = {}
    fps_list = []
    frame_counts_list = []
    resolutions_list = []
    
    print("\n--- Loading Videos ---")
    for camera_id, video_path in enumerate(video_files):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = (width, height)
        
        captures[camera_id] = cap
        fps_list.append(fps)
        frame_counts_list.append(frame_count)
        resolutions_list.append(resolution)
        
        print(f"Camera {camera_id}: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {frame_count}")
    
    metadata = {
        'fps': fps_list,
        'frame_counts': frame_counts_list,
        'resolutions': resolutions_list
    }
    
    return captures, metadata


def read_synchronized_frames(captures: Dict[int, cv2.VideoCapture]) -> Tuple[Dict[int, any], bool]:
    """
    Read one frame from each active camera.
    
    Args:
        captures: dict mapping camera_id -> VideoCapture
        
    Returns:
        Tuple of (frames_dict, all_finished) where:
        - frames_dict: camera_id -> frame (only successful reads)
        - all_finished: True if no more frames from any camera
    """
    frames_dict = {}
    any_success = False
    
    for camera_id, cap in captures.items():
        ret, frame = cap.read()
        if ret:
            frames_dict[camera_id] = frame
            any_success = True
    
    all_finished = not any_success
    return frames_dict, all_finished


def main():
    """Main entry point for the multi-camera person tracking system."""
    print("=== 2D Event Map System ===")
    
    try:
        # Validate video files first
        print("\n[PRE-CHECK] Validating video files...")
        try:
            validate_video_files(VIDEO_FILES)
            print("✓ All video files validated")
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            print("Please check your video file paths in config.py")
            sys.exit(1)
        
        # Phase 1: Load videos
        print("\n[PHASE 1] Loading videos...")
        captures, metadata = load_videos(VIDEO_FILES)
        print(f"Loaded {len(captures)} camera(s)")
        
        # Phase 2: Run calibration
        print("\n[PHASE 2] Starting calibration...")
        print("Instructions:")
        print("  - Click 4 points on each camera view to define the floor area")
        print("  - Points should form a quadrilateral (e.g., corners of a room)")
        print("  - Press SPACE to move to next camera")
        print("  - Close window to cancel calibration")
        
        try:
            calibration_data = calibrate_cameras(VIDEO_FILES, CANVAS_WIDTH, CANVAS_HEIGHT)
            print("✓ Calibration complete!")
        except RuntimeError as e:
            error_msg = str(e)
            if "cancelled" in error_msg.lower():
                print(f"\nCalibration cancelled by user")
                for cap in captures.values():
                    cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            else:
                print(f"\nCalibration error: {e}")
                for cap in captures.values():
                    cap.release()
                cv2.destroyAllWindows()
                sys.exit(1)
        
        # Reset video captures to frame 0
        print("\n[PHASE 3] Resetting videos to start...")
        for cap in captures.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Phase 3: Initialize PersonDetector
        print("\n[PHASE 4] Initializing person detector...")
        detector = PersonDetector()
        
        # Phase 4: Initialize CoordinateTransformer
        print("\n[PHASE 5] Initializing coordinate transformer...")
        transformer = CoordinateTransformer(calibration_data)
        
        # Phase 5: Initialize MapVisualizer
        print("\n[PHASE 6] Initializing map visualizer...")
        visualizer = MapVisualizer(CANVAS_WIDTH, CANVAS_HEIGHT)
        
        # Get grid dimensions
        grid_width, grid_height = calculate_grid_dimensions(len(captures))
        
        # Main processing loop
        print("\n[PHASE 7] Starting video processing...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'p' to pause/resume")
        print()
        
        frame_count = 0
        paused = False
        
        while True:
            # Handle pause
            if paused:
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '):  # SPACE to resume
                    paused = False
                    print("Resumed")
                elif key == ord('q'):
                    break
                continue
            
            # Read synchronized frames
            frames_dict, all_finished = read_synchronized_frames(captures)
            
            if all_finished:
                print("\nAll videos finished")
                break
            
            # Detect persons in all frames
            detections_per_camera = {}
            total_detections = 0
            
            for camera_id, frame in frames_dict.items():
                try:
                    detections = detector.detect_persons(frame)
                    detections_per_camera[camera_id] = detections
                    total_detections += len(detections)
                except Exception as e:
                    print(f"Warning: Detection failed for camera {camera_id}: {e}")
                    detections_per_camera[camera_id] = []
            
            # Transform detections to canvas coordinates
            canvas_detections = transformer.transform_detections(detections_per_camera)
            
            # Render and display 2D map
            map_frame = visualizer.render_frame(
                canvas_detections,
                show_grid=True,
                grid_rows=grid_width,
                grid_cols=grid_height
            )
            
            cv2.imshow("2D Event Map", map_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('p'):
                paused = True
                print("Paused (press SPACE to resume)")
            
            # Increment frame counter
            frame_count += 1
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {len(frames_dict)} cameras active, {total_detections} persons detected")
        
        # Cleanup
        print("\n[CLEANUP] Releasing resources...")
        for cap in captures.values():
            cap.release()
        visualizer.close()
        cv2.destroyAllWindows()
        
        print("Done!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if 'captures' in locals():
            for cap in captures.values():
                cap.release()
        if 'visualizer' in locals():
            visualizer.close()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\nCleaning up...")
        if 'captures' in locals():
            for cap in captures.values():
                cap.release()
        if 'visualizer' in locals():
            visualizer.close()
        cv2.destroyAllWindows()
        sys.exit(1)


if __name__ == "__main__":
    main()
