# AI Agent Prompts for 2D Event Map Implementation

## Instructions for Using These Prompts

Each prompt below corresponds to one implementation phase. Use them sequentially:

1. Start a new conversation with the AI agent
2. Copy and paste the prompt for the current phase
3. Let the agent implement the code
4. Test the implementation using the provided test instructions
5. Once tests pass, move to the next phase in a NEW conversation
6. Repeat until all phases complete

**Important Notes:**
- Do NOT ask the agent to install packages - venv already has opencv and ultralytics
- Each phase builds on previous phases - ensure previous phase works before proceeding
- Keep the implementation plan (IMPLEMENTATION_PLAN.md) open for reference
- Test thoroughly before moving to next phase

---

## PHASE 1 PROMPT: Project Setup & Configuration

```
I'm building a multi-camera person tracking system that displays detections on a 2D floor map. This is PHASE 1 of the implementation.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Python venv already set up with opencv-python, numpy, and ultralytics installed
- DO NOT run pip install commands
- Reference implementation plan: IMPLEMENTATION_PLAN.md (already exists in project)

PHASE 1 OBJECTIVES:
Create project structure, configuration files, and utility functions.

TASKS TO COMPLETE:

1. Create config.py with these configurations:
   - VIDEO_FILES: List of video file paths (use placeholder paths for now, like "/path/to/camera1.mp4")
   - DETECTION_CONFIDENCE_THRESHOLD = 0.5
   - YOLO_MODEL = "yolov8n.pt"
   - CANVAS_WIDTH = 1200
   - CANVAS_HEIGHT = 1200
   - CANVAS_BACKGROUND_COLOR = (50, 50, 50)
   - DOT_COLOR = (0, 255, 0)
   - DOT_RADIUS = 8
   - DOT_THICKNESS = -1
   - POINT_COLOR = (0, 0, 255)
   - POINT_RADIUS = 5

2. Create utils.py with these functions:
   - validate_video_files(video_paths): Check if video files exist and can be opened
   - calculate_grid_dimensions(num_cameras): Calculate grid rows/cols (e.g., 1 camera->1x1, 2->2x1, 3-4->2x2, 5-6->3x2)
   - validate_calibration_points(points): Ensure 4 points form valid quadrilateral

3. Create empty module files:
   - calibration.py (with module docstring explaining purpose)
   - detector.py (with module docstring)
   - transformer.py (with module docstring)
   - visualizer.py (with module docstring)
   - main.py (with module docstring)

4. Create tests/test_phase1.py that:
   - Imports all modules (verify no syntax errors)
   - Prints all config values
   - Tests calculate_grid_dimensions with various camera counts (1, 2, 4, 6, 9)
   - Prints results

IMPLEMENTATION NOTES:
- Use proper error handling in validation functions
- Add clear docstrings to all functions
- Follow PEP 8 style guidelines
- Make code modular and readable

After implementation, I will test by running:
```bash
python tests/test_phase1.py
```

Please implement Phase 1 now.
```

---

## PHASE 2 PROMPT: Calibration Module

```
I'm continuing the multi-camera person tracking system. This is PHASE 2 of the implementation.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Phase 1 completed: config.py, utils.py, and empty modules exist
- Python venv has opencv-python, numpy, ultralytics installed
- Reference: IMPLEMENTATION_PLAN.md sections for Phase 2

PHASE 2 OBJECTIVES:
Implement calibration module for floor point selection and homography computation.

TASKS TO COMPLETE:

1. Implement calibration.py with:

   A. Class PointSelector:
      - __init__(self, frame, camera_id)
      - select_points(self): Display frame, capture 4 mouse clicks for floor corners
        * Show instructions in window title or on frame
        * Draw red circles at clicked points
        * Draw lines connecting points
        * Show point count (e.g., "Points: 2/4")
        * Keys: 'r' to reset, 'c' to confirm (only when 4 points), 'q' to quit
        * Return list of 4 (x, y) tuples
        * Add error handling for invalid points

   B. Function calculate_grid_dimensions(num_cameras):
      - Calculate rows and cols for grid layout
      - Return (rows, cols) tuple

   C. Function compute_homography(source_points, destination_points):
      - Use cv2.getPerspectiveTransform()
      - Validate matrix is not None and not singular
      - Return 3x3 homography matrix

   D. Function calibrate_cameras(video_files, canvas_width, canvas_height):
      - Load first frame from each video
      - For each camera: run PointSelector to get 4 floor points
      - Calculate grid layout (auto-arrange cameras in grid)
      - Calculate destination rectangle corners for each camera based on grid cell
      - Compute homography for each camera
      - Return calibration_data dict with structure:
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

2. Create tests/test_calibration.py that:
   - Tests PointSelector with a sample frame (manual test - user clicks points)
   - Tests compute_homography with sample points
   - Tests homography transformation on sample points
   - Prints all matrices and results for verification

IMPLEMENTATION NOTES:
- For grid layout: use formula rows = ceil(sqrt(n)), cols = ceil(n/rows)
- Destination points for grid cell (row, col):
  * cell_w = canvas_width / cols, cell_h = canvas_height / rows
  * top_left = (col * cell_w, row * cell_h)
  * top_right = ((col+1) * cell_w, row * cell_h)
  * bottom_right = ((col+1) * cell_w, (row+1) * cell_h)
  * bottom_left = (col * cell_w, (row+1) * cell_h)
- Add comprehensive error handling
- Use cv2.setMouseCallback for point selection

Before asking me to test, show me the key functions you implemented so I can review.

After review, I will test manually by running the test script with sample video frames.

Please implement Phase 2 now.
```

---

## PHASE 2B PROMPT: Interactive Grid Arrangement UI

```
I'm continuing the multi-camera person tracking system. This is PHASE 2B - an enhancement to the calibration module.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Phase 2 completed: calibration.py works with auto-arrangement
- Currently: cameras are auto-arranged in grid (camera 0->cell 0,0, camera 1->cell 0,1, etc.)
- Enhancement needed: Interactive UI where user can see and rearrange camera coverage areas
- Reference: IMPLEMENTATION_PLAN.md sections on "Grid-Based Camera Layout"

PHASE 2B OBJECTIVES:
Add interactive grid arrangement UI so users can visually position cameras to match actual room layout.

CURRENT BEHAVIOR:
- calibrate_cameras() auto-arranges cameras in row-major order
- Works but doesn't let user customize spatial arrangement

DESIRED BEHAVIOR:
After point selection, show a canvas where user can:
1. See all camera coverage areas as labeled rectangles
2. Drag/rearrange them to match actual room layout
3. Confirm arrangement before computing homographies

TASKS TO COMPLETE:

1. Add new class to calibration.py:

   Class GridArrangementUI:
      - __init__(self, num_cameras, camera_frames, canvas_width, canvas_height):
        * Store camera count, canvas dimensions
        * Store thumbnail of each camera's first frame
        * Initialize with default grid positions
        
      - arrange_cameras(self):
        * Display canvas with current grid layout
        * Show each camera as labeled rectangle with thumbnail
        * Show camera ID labels (Camera 0, Camera 1, etc.)
        * Allow drag-and-drop to swap camera positions
        * OR allow keyboard input to reassign positions
        * Keys: arrow keys to select camera, WASD to move, 'c' to confirm, 'r' to reset
        * Return: dict mapping camera_id -> (grid_row, grid_col)

   Alternative Simpler Approach (RECOMMENDED):
      - arrange_cameras_simple(self):
        * Display canvas showing current auto-arrangement with labels
        * Show instruction: "Cameras arranged automatically. Press 'c' to confirm, 'e' to edit"
        * If 'e' pressed: show prompt asking user to input custom arrangement
        * Input format: "camera_id:row,col camera_id:row,col" (e.g., "0:0,1 1:0,0 2:1,0")
        * Validate input and update arrangement
        * Display updated arrangement and confirm
        * Return: dict mapping camera_id -> (grid_row, grid_col)

2. Modify calibrate_cameras() function:
   - After point selection for all cameras
   - Before computing homographies
   - Create GridArrangementUI instance
   - Call arrange_cameras_simple() to get camera positions
   - Use returned positions instead of auto-calculated ones
   - Compute destination points based on user-specified grid cells
   - Rest of the flow remains the same

3. Update tests/test_calibration.py:
   - Add test for GridArrangementUI
   - Test default arrangement display
   - Test manual arrangement (if user chooses to edit)
   - Verify grid cell assignments are correct

IMPLEMENTATION NOTES:
- Keep it simple: text-based input is easier than drag-and-drop
- Default arrangement should be shown visually (draw grid with camera labels)
- User can accept default or customize
- Validation: ensure each grid cell has at most one camera
- Validation: ensure all camera IDs are assigned
- Draw camera thumbnails (resized) in their grid cells for visual reference

EXPECTED WORKFLOW:
1. User selects floor points for each camera (existing behavior)
2. NEW: System shows grid layout with camera positions
3. NEW: User confirms or customizes arrangement
4. System computes homographies based on final arrangement (existing code)

VISUAL LAYOUT EXAMPLE:
```
Canvas with 2x2 grid for 3 cameras:
+------------------------+------------------------+
|      Camera 0          |      Camera 1          |
|   [thumbnail]          |   [thumbnail]          |
|                        |                        |
+------------------------+------------------------+
|      Camera 2          |        Empty           |
|   [thumbnail]          |                        |
|                        |                        |
+------------------------+------------------------+

Instructions: Press 'c' to confirm, 'e' to edit positions
```

IMPORTANT NOTES:
- Don't break existing functionality
- If user just presses 'c' without editing, use auto-arrangement
- This is an enhancement, not a replacement
- Keep the implementation simple - focus on functionality over fancy UI

After implementation, show me the GridArrangementUI class and updated calibrate_cameras function.

Then I will test the interactive arrangement workflow.

Please implement Phase 2B now.
```

---

## PHASE 3 PROMPT: Detection Module

```
I'm continuing the multi-camera person tracking system. This is PHASE 3 of the implementation.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Phase 1 & 2 completed: config, utils, calibration modules working
- Python venv has ultralytics (YOLOv8) installed
- Reference: IMPLEMENTATION_PLAN.md sections for Phase 3

PHASE 3 OBJECTIVES:
Implement person detection using YOLOv8.

TASKS TO COMPLETE:

1. Implement detector.py with:

   A. Class PersonDetector:
      - __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        * Load YOLO model using: self.model = YOLO(model_name)
        * Store confidence threshold
        * Set person_class_id = 0 (COCO dataset person class)
        * Add error handling for model loading

      - detect_persons(self, frame):
        * Run inference: results = self.model(frame, verbose=False)
        * Filter for person class (cls == 0) and confidence >= threshold
        * For each detection, extract bounding box [x1, y1, x2, y2]
        * Calculate foot position: (center_x, bottom_y) = ((x1+x2)/2, y2)
        * Return list of dicts with structure:
          [
              {
                  'bbox': [x1, y1, x2, y2],
                  'confidence': float,
                  'foot_position': (x, y)
              },
              ...
          ]

      - detect_persons_batch(self, frames):
        * Accept list/dict of frames
        * Process each frame
        * Return corresponding list/dict of detection lists

2. Create tests/test_detector.py that:
   - Initializes PersonDetector with yolov8n.pt
   - Loads a test video frame (or creates synthetic frame with rectangle to test)
   - Runs detection and prints results
   - Draws bounding boxes and foot points on frame
   - Displays annotated frame for visual verification
   - Tests batch processing with multiple frames

IMPLEMENTATION NOTES:
- YOLO results structure: results[0].boxes contains detection boxes
- Box attributes: box.xyxy[0] = coordinates, box.conf = confidence, box.cls = class
- Need to convert tensors to numpy: use .cpu().numpy() or float()
- First run will download YOLOv8 weights automatically (this is okay)
- Add try-except around inference in case frame is invalid

EXPECTED BEHAVIOR:
- On frames with people: should return detections with foot positions
- On empty frames: should return empty list
- Confidence threshold filters low-confidence detections

Before implementing, confirm you understand the YOLOv8 API from ultralytics library.

After implementation, show me the detect_persons method so I can review.

Then I will test with sample video frames.

Please implement Phase 3 now.
```

---

## PHASE 4 PROMPT: Transformation Module

```
I'm continuing the multi-camera person tracking system. This is PHASE 4 of the implementation.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Phases 1-3 completed: config, utils, calibration, detection modules working
- We have homography matrices from calibration
- We have foot positions from detection
- Reference: IMPLEMENTATION_PLAN.md sections for Phase 4

PHASE 4 OBJECTIVES:
Transform foot positions from camera coordinates to canvas coordinates using homography.

TASKS TO COMPLETE:

1. Implement transformer.py with:

   A. Function transform_point(point, homography_matrix):
      - Takes single (x, y) tuple and 3x3 homography matrix
      - Use cv2.perspectiveTransform() to transform point
      - Format: point_array = np.array([[point]], dtype=np.float32)
      - Return transformed (x, y) tuple as floats

   B. Function transform_points(points, homography_matrix):
      - Takes list of (x, y) tuples and 3x3 homography matrix
      - Transform all points in batch using cv2.perspectiveTransform()
      - Return list of transformed (x, y) tuples
      - Handle empty list case

   C. Function validate_canvas_coordinates(points, canvas_width, canvas_height):
      - Filter points to only those within canvas bounds (0 <= x <= width, 0 <= y <= height)
      - Return (valid_points, invalid_count)
      - valid_points: list of points within bounds
      - invalid_count: number of out-of-bounds points

   D. Class CoordinateTransformer:
      - __init__(self, calibration_data):
        * Extract canvas dimensions
        * Build dict mapping camera_id -> homography_matrix
        * Store camera_homographies dict

      - transform_detections(self, detections_per_camera):
        * Input: dict mapping camera_id -> list of detection dicts
        * For each camera, extract foot_position from each detection
        * Transform foot positions using that camera's homography
        * Validate transformed points are within canvas
        * Aggregate all valid points into single list
        * Return list of (x, y) tuples representing positions on canvas

2. Create tests/test_transformer.py that:
   - Tests transform_point with identity matrix (should return same point)
   - Tests transform_points with sample homography
   - Tests validate_canvas_coordinates with points inside and outside bounds
   - Creates mock calibration data with 2 cameras
   - Creates mock detections (foot positions) for each camera
   - Tests CoordinateTransformer.transform_detections()
   - Prints all transformed coordinates
   - Verifies results are reasonable

IMPLEMENTATION NOTES:
- cv2.perspectiveTransform expects shape (1, N, 2) for N points
- Homography matrices come from calibration phase
- Points outside canvas are expected (person outside calibrated floor area)
- Coordinate system: (0,0) is top-left, (width, height) is bottom-right

EXPECTED BEHAVIOR:
- Points in camera view → transformed to canvas coordinates
- Points maintain spatial relationships after transformation
- Out-of-bounds points filtered out without errors

After implementation, show me the CoordinateTransformer class so I can review.

Then I will test with mock data and verify transformations are correct.

Please implement Phase 4 now.
```

---

## PHASE 5 PROMPT: Visualization Module

```
I'm continuing the multi-camera person tracking system. This is PHASE 5 of the implementation.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Phases 1-4 completed: config, utils, calibration, detection, transformation modules working
- We can now transform detections to canvas coordinates
- Reference: IMPLEMENTATION_PLAN.md sections for Phase 5

PHASE 5 OBJECTIVES:
Create 2D map visualization that displays detection points.

TASKS TO COMPLETE:

1. Implement visualizer.py with:

   A. Class MapVisualizer:
      - __init__(self, canvas_width, canvas_height, background_color=(50, 50, 50)):
        * Store canvas dimensions and background color
        * Set window_name = "2D Event Map"
        * Create OpenCV window: cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        * Resize window to canvas dimensions

      - create_blank_canvas(self):
        * Create numpy array filled with background color
        * Shape: (canvas_height, canvas_width, 3)
        * Dtype: np.uint8
        * Return canvas

      - draw_grid(self, canvas, rows, cols, color=(80, 80, 80)):
        * Draw grid lines on canvas
        * Vertical lines at column boundaries
        * Horizontal lines at row boundaries
        * Use cv2.line()

      - draw_detections(self, canvas, points, color=(0, 255, 0), radius=8, thickness=-1):
        * Draw filled circles at each point
        * Points: list of (x, y) tuples
        * Use cv2.circle()
        * Convert coordinates to int before drawing

      - render_frame(self, detection_points, show_grid=False, grid_rows=2, grid_cols=2):
        * Create blank canvas
        * Optionally draw grid if show_grid=True
        * Draw detection points
        * Return rendered canvas

      - show(self, canvas, wait_key=1):
        * Display canvas using cv2.imshow()
        * Wait for key press (wait_key milliseconds)
        * Return key code (or -1 if no key)

      - close(self):
        * Destroy the visualization window
        * Call cv2.destroyWindow(window_name)

2. Create tests/test_visualizer.py that:
   - Initializes MapVisualizer
   - Creates and displays blank canvas (verify size and color)
   - Tests draw_grid (visual verification)
   - Tests drawing points at various positions:
     * Center: (600, 600)
     * Corners: (50, 50), (1150, 50), (1150, 1150), (50, 1150)
     * Random: (300, 400), (800, 700)
   - Tests render_frame with multiple points and grid
   - Creates simple animation: moving point across canvas (30 frames)
   - Tests keyboard input (press 'q' to quit animation)

IMPLEMENTATION NOTES:
- OpenCV color format is BGR (not RGB)
- Canvas coordinates: (0, 0) is top-left
- cv2.circle center must be int tuple
- wait_key(1) = 1ms delay (for real-time feel)
- wait_key(0) = wait indefinitely until key press

EXPECTED BEHAVIOR:
- Canvas displays with correct size and background color
- Grid lines visible and aligned to grid dimensions
- Dots appear at specified coordinates
- Animation runs smoothly
- Window responds to keyboard input

After implementation, show me the render_frame and show methods so I can review.

Then I will run tests to verify visualization works correctly.

Please implement Phase 5 now.
```

---

## PHASE 6 PROMPT: Main Orchestrator

```
I'm continuing the multi-camera person tracking system. This is PHASE 6 of the implementation - INTEGRATION.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Phases 1-5 completed: All modules implemented and tested individually
- Now we integrate everything into main processing pipeline
- Reference: IMPLEMENTATION_PLAN.md sections for Phase 6

PHASE 6 OBJECTIVES:
Integrate all modules into complete system with video processing loop.

TASKS TO COMPLETE:

1. Update config.py:
   - Change VIDEO_FILES to actual video file paths (I will provide these paths)
   - For now, use placeholder list with 2-3 paths like:
     VIDEO_FILES = [
         "/home/ezio/Documents/work/vis-tracking/videos/camera1.mp4",
         "/home/ezio/Documents/work/vis-tracking/videos/camera2.mp4",
     ]

2. Implement main.py with:

   A. Function load_videos(video_files):
      - Create cv2.VideoCapture for each video
      - Return dict: camera_id -> VideoCapture
      - Also return metadata dict with fps, frame_count, resolution per camera
      - Print info for each loaded video
      - Raise error if video cannot be opened

   B. Function read_synchronized_frames(captures):
      - Read one frame from each active camera
      - Return (frames_dict, all_finished)
      - frames_dict: camera_id -> frame (only cameras that successfully read)
      - all_finished: True if no more frames from any camera

   C. Function main():
      - Print banner: "=== 2D Event Map System ==="
      - Phase 1: Load videos using load_videos()
      - Phase 2: Run calibration using calibrate_cameras()
      - Reset video captures to frame 0
      - Phase 3: Initialize PersonDetector
      - Phase 4: Initialize CoordinateTransformer with calibration data
      - Phase 5: Initialize MapVisualizer
      - Get grid dimensions using calculate_grid_dimensions()
      - Main processing loop:
        * Read synchronized frames
        * Check if all finished -> break
        * Detect persons in all frames (create detections_per_camera dict)
        * Transform detections to canvas coordinates
        * Render and display 2D map with detections
        * Handle keyboard input:
          - 'q': quit
          - 'p': pause (wait for SPACE to resume)
        * Print progress every 30 frames (frame count, active cameras, total detections)
      - Cleanup: release all captures, close visualizer, destroy windows
      - Print "Done"

   D. Add error handling:
      - Wrap main() in try-except
      - Handle KeyboardInterrupt (Ctrl+C)
      - Handle general exceptions with traceback
      - Always cleanup on exit

   E. Add if __name__ == "__main__": main()

IMPLEMENTATION NOTES:
- Import all modules: config, utils, calibration, detector, transformer, visualizer
- Use descriptive print statements for user feedback
- Progress format: "Frame 30: 2 cameras active, 3 persons detected"
- Pause logic: set paused=True, loop with cv2.waitKey(10) until SPACE pressed
- Grid dimensions needed for visualizer.render_frame(show_grid=True)

EXPECTED WORKFLOW:
1. User runs: python main.py
2. Videos load and info printed
3. Calibration UI appears (user clicks points for each camera)
4. Detection starts, 2D map window shows
5. Dots appear on map showing detected persons
6. User can pause ('p') or quit ('q')
7. When videos end, program exits cleanly

SPECIAL INSTRUCTIONS:
- Don't worry about video files not existing yet - I'll create test videos
- Make sure calibration can be cancelled (handle window close)
- Print clear instructions for calibration step
- Add helpful progress messages

After implementation, show me the main() function so I can review the flow.

Then I will create test videos and run the complete system.

Please implement Phase 6 now.
```

---

## PHASE 7 PROMPT: Error Handling & Validation

```
I'm continuing the multi-camera person tracking system. This is PHASE 7 of the implementation - POLISH.

PROJECT CONTEXT:
- Working directory: /home/ezio/Documents/work/vis-tracking
- Phases 1-6 completed: Full system working end-to-end
- Now we add comprehensive error handling and validation
- Reference: IMPLEMENTATION_PLAN.md sections for Phase 7

PHASE 7 OBJECTIVES:
Add robust error handling, input validation, and graceful failure recovery.

TASKS TO COMPLETE:

1. Enhance utils.py with validation functions:
   - validate_point_count(points, expected=4): Raise error if wrong count
   - validate_points_not_collinear(points): Check points form valid quadrilateral using cv2.contourArea()
   - validate_homography_matrix(matrix): Check not None and not singular (determinant check)
   - Enhance validate_video_files(): Check files exist AND can be opened

2. Add error handling to calibration.py:
   - In PointSelector.select_points():
     * Validate 4 points collected
     * Call validate_points_not_collinear()
     * Handle window close (user cancels) -> return None
     * Wrap in try-except, raise RuntimeError with camera ID on failure
   
   - In compute_homography():
     * Call validate_homography_matrix()
     * Wrap in try-except, raise RuntimeError on failure
   
   - In calibrate_cameras():
     * Check if point selection returned None (user cancelled) -> raise error
     * Add try-except for each camera calibration step

3. Add error handling to detector.py:
   - In PersonDetector.__init__():
     * Wrap model loading in try-except
     * Raise RuntimeError if model fails to load
   
   - In detect_persons():
     * Wrap inference in try-except
     * On error: print warning and return empty list (don't crash)

4. Add error handling to main.py:
   - Before calibration: call validate_video_files(VIDEO_FILES)
   - Handle calibration failure (print clear message and exit)
   - Handle detection errors gracefully (skip frame, continue)
   - Improve KeyboardInterrupt handler (print clean message)

5. Create tests/test_errors.py that tests:
   - Invalid video path -> expect clear error
   - Only 3 points in calibration -> expect validation error
   - Collinear points -> expect validation error
   - Empty detections (no people) -> no crash, empty map
   - Keyboard interrupt -> clean exit

IMPLEMENTATION NOTES:
- Error messages should be clear and actionable
- Validation should happen early (fail fast)
- Detection errors should not crash entire system
- Always cleanup resources on any exit path
- Use specific exception types (ValueError, FileNotFoundError, RuntimeError)

EXPECTED IMPROVEMENTS:
- Clear error messages guide user to fix issues
- System doesn't crash on invalid input
- Partial failures handled gracefully (e.g., one camera fails, others continue)
- Resources always cleaned up

After implementation, show me the key validation functions and error handling patterns.

Then I will test various error scenarios to verify robustness.

Please implement Phase 7 now.
```

---

## TESTING INSTRUCTIONS

After each phase implementation:

### Phase 1 Testing
```bash
cd /home/ezio/Documents/work/vis-tracking
python tests/test_phase1.py
```
Verify: All imports work, config prints correctly, grid calculations correct

### Phase 2 Testing
```bash
python tests/test_calibration.py
```
Manually: Click 4 points when frame appears
Verify: Points collected, homography computed, transformation works

### Phase 3 Testing
```bash
python tests/test_detector.py
```
Verify: Model loads, detections found (if people in frame), bounding boxes correct

### Phase 4 Testing
```bash
python tests/test_transformer.py
```
Verify: Points transform correctly, out-of-bounds filtered, multi-camera aggregation works

### Phase 5 Testing
```bash
python tests/test_visualizer.py
```
Verify: Canvas displays, grid visible, dots at correct positions, animation smooth

### Phase 6 Testing
```bash
# First, create test videos directory and add sample videos
mkdir -p videos
# Copy test videos to videos/ directory

# Update config.py with actual video paths
# Then run:
python main.py
```
Manually: Complete calibration, watch detections on map
Verify: Full pipeline works, detections appear correctly positioned

### Phase 7 Testing
```bash
python tests/test_errors.py

# Also test manually:
# - Run with invalid video path
# - Cancel calibration (close window)
# - Press Ctrl+C during processing
```
Verify: Errors handled gracefully, clear messages, clean exit

---

## TROUBLESHOOTING GUIDE

### Common Issues

**Issue: "Cannot import module"**
- Check all files created in correct directory
- Verify no syntax errors in imported module
- Check file names match exactly

**Issue: "Video file not found"**
- Update VIDEO_FILES in config.py with actual paths
- Ensure videos exist and are readable

**Issue: "YOLO model download timeout"**
- Weights auto-download on first run
- Requires internet connection
- Downloads to ~/.ultralytics/

**Issue: "Homography transformation looks wrong"**
- Check point order is consistent (clockwise or counter-clockwise)
- Verify source points are actually floor corners in camera view
- Check destination rectangles don't overlap

**Issue: "No detections found"**
- Check video has visible people
- Lower confidence threshold (try 0.3)
- Verify YOLO model loaded correctly

**Issue: "Window doesn't respond"**
- Add cv2.waitKey() call in loop
- Check window hasn't been closed manually

---

## SUCCESS CRITERIA

### Phase 1
✅ All modules import without errors
✅ Config values print correctly
✅ Grid calculations work for various camera counts

### Phase 2
✅ Can click 4 points on camera frame
✅ Homography matrix computed (3x3 array)
✅ Test transformation produces reasonable coordinates

### Phase 3
✅ YOLO model loads successfully
✅ Detections found in frames with people
✅ Foot positions at bottom-center of bounding boxes

### Phase 4
✅ Single point transforms correctly
✅ Batch transformation works
✅ Out-of-bounds points filtered
✅ Multi-camera detections aggregated

### Phase 5
✅ Canvas displays with correct size
✅ Grid lines visible
✅ Dots appear at specified coordinates
✅ Animation runs smoothly

### Phase 6
✅ Videos load successfully
✅ Calibration completes
✅ Detections appear on 2D map
✅ Dots positioned correctly relative to camera coverage
✅ Can pause and quit
✅ Handles videos finishing at different times

### Phase 7
✅ Invalid inputs rejected with clear errors
✅ Calibration cancellation handled
✅ Detection errors don't crash system
✅ Ctrl+C exits cleanly
✅ Resources always cleaned up

---

## FINAL CHECKLIST

Before considering the project complete, verify:

- [ ] All 7 phases implemented
- [ ] All test scripts run successfully
- [ ] Full system runs end-to-end with real videos
- [ ] Calibration is intuitive and works correctly
- [ ] Detections appear in correct positions on map
- [ ] Multiple cameras work simultaneously
- [ ] Error handling prevents crashes
- [ ] Code is clean and well-documented
- [ ] Can process full-length videos without issues

**Congratulations! You've built a multi-camera 2D event mapping system!** 🎉
