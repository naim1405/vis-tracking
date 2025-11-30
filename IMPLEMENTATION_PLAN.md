# 2D Event Map System - Comprehensive Implementation Plan

## Project Overview

### Goal
Create a system that processes multiple camera video feeds to track people and display their positions on a unified 2D floor map in real-time.

### Core Concept
- Multiple cameras cover different areas of a room
- Person detection identifies people in each camera view
- Homography transformation maps camera pixels to 2D floor coordinates
- Combined visualization shows all detected persons on a single top-down map

---

## Architecture Rationale

### Key Design Decisions

#### 1. **Calibration Without Persistence (Phase 1)**
**Decision:** Run calibration every time the program starts, without saving configuration
**Rationale:**
- Simplifies initial implementation
- Avoids file I/O and serialization complexity
- Easier to iterate and test during development
- Can add persistence in Phase 2 without changing core logic

#### 2. **Canvas-Based Coordinate System**
**Decision:** Use canvas pixel coordinates as the map coordinate system
**Rationale:**
- Direct mapping: camera pixels → canvas pixels (single transformation)
- Simpler than abstract coordinate systems requiring two transformations
- Tightly coupled to visualization, reducing complexity
- Sufficient for non-scaled visualization requirements

#### 3. **Homography Transformation**
**Decision:** Use 4-point homography for perspective correction
**Rationale:**
- Minimum viable calibration (4 points per camera)
- OpenCV provides robust homography computation
- Handles perspective distortion effectively
- No need for complex camera intrinsics or 3D reconstruction

#### 4. **Grid-Based Camera Layout**
**Decision:** Automatic grid generation based on camera count
**Rationale:**
- User arranges cameras spatially on a grid
- Natural mental model for room coverage
- Simplifies destination point calculation
- Easy to visualize and validate during calibration

#### 5. **Synchronized Frame Processing**
**Decision:** Process all videos frame-by-frame in sync
**Rationale:**
- Maintains temporal consistency across cameras
- Simpler logic than asynchronous processing
- Suitable for file-based input (not real-time streaming)
- Handles different video lengths gracefully (ignore finished cameras)

#### 6. **YOLOv8 for Detection**
**Decision:** Use YOLOv8 via ultralytics library
**Rationale:**
- State-of-art accuracy for person detection
- Easy to use (one-line installation and inference)
- Pre-trained weights readily available
- Good performance/speed tradeoff
- Active maintenance and support

#### 7. **Modular Architecture**
**Decision:** Separate concerns into distinct modules
**Rationale:**
- Easier to test individual components
- Clear separation of responsibilities
- Facilitates incremental development
- Makes code maintainable and extensible
- Allows parallel development of modules

#### 8. **OpenCV for UI**
**Decision:** Use OpenCV for both calibration and visualization
**Rationale:**
- Minimal dependencies (already needed for video/image processing)
- Sufficient for basic UI needs
- No framework overhead (PyQt/Tkinter)
- Simple event handling for mouse clicks
- Fast to implement

---

## Project Structure

```
vis-tracking/
├── config.py           # Configuration: video paths, constants, parameters
├── calibration.py      # Calibration UI and homography computation
├── detector.py         # YOLOv8 person detection
├── transformer.py      # Coordinate transformation logic
├── visualizer.py       # 2D map rendering
├── utils.py           # Helper functions and validation
├── main.py            # Main orchestrator
├── tests/
│   ├── test_calibration.py
│   ├── test_detector.py
│   ├── test_transformer.py
│   └── test_visualizer.py
├── requirements.txt
└── IMPLEMENTATION_PLAN.md (this file)
```

---

## Implementation Phases

---

## PHASE 1: Project Setup & Configuration

### Objectives
- Establish project structure
- Define configuration parameters
- Set up dependencies
- Verify video files are accessible

### Tasks

#### Task 1.1: Create Project Structure
- Create all module files (config.py, calibration.py, detector.py, transformer.py, visualizer.py, utils.py, main.py)
- Create tests/ directory with test files
- Create requirements.txt

#### Task 1.2: Define Configuration (config.py)
**What to implement:**
```python
# Video file paths - list of video file paths
VIDEO_FILES = [
    "/path/to/camera1.mp4",
    "/path/to/camera2.mp4",
    "/path/to/camera3.mp4",
    "/path/to/camera4.mp4"
]

# Detection parameters
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection
YOLO_MODEL = "yolov8n.pt"  # YOLOv8 nano model (fastest)

# Canvas/Map parameters
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1200
CANVAS_BACKGROUND_COLOR = (50, 50, 50)  # Dark gray

# Visualization parameters
DOT_COLOR = (0, 255, 0)  # Green dots for detected persons
DOT_RADIUS = 8
DOT_THICKNESS = -1  # Filled circle

# Calibration parameters
POINT_COLOR = (0, 0, 255)  # Red for calibration points
POINT_RADIUS = 5
```

**Why these values:**
- YOLOv8n (nano): Fastest variant, good for real-time processing
- Confidence 0.5: Balance between false positives and false negatives
- Canvas 1200x1200: Large enough for detail, not too large for display
- Green dots: High visibility on dark background

#### Task 1.3: Create requirements.txt
**Dependencies:**
```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
```

**Why these:**
- opencv-python: Video I/O, image processing, GUI, homography
- numpy: Numerical operations, array handling
- ultralytics: YOLOv8 implementation and pre-trained models

#### Task 1.4: Create Utility Functions (utils.py)
**What to implement:**
```python
def validate_video_files(video_paths):
    """Verify all video files exist and can be opened"""
    
def calculate_grid_dimensions(num_cameras):
    """Calculate grid rows/cols based on camera count"""
    # For n cameras: try to create square grid
    # 1 camera: 1x1, 2 cameras: 2x1, 3 cameras: 2x2, 4 cameras: 2x2
    # 5-6 cameras: 3x2, etc.
    
def validate_calibration_points(points):
    """Ensure points form valid quadrilateral (4 points, non-collinear)"""
```

**Rationale:**
- Centralized validation logic
- Reusable across modules
- Clear error messages

### Testing Phase 1

#### Test Script: test_phase1.py
**What to test:**
1. Import all modules successfully (no syntax errors)
2. Print configuration values
3. Validate video files exist
4. Open each video and read first frame
5. Display first frames to verify video loading

**Success Criteria:**
- All imports work without errors
- Configuration prints correctly
- All video files load successfully
- First frames display in OpenCV windows
- No exceptions raised

**How to test:**
```bash
python test_phase1.py
```

**Expected output:**
```
✓ All modules imported successfully
✓ Configuration loaded
✓ Found 4 video files
✓ camera1.mp4 - Resolution: 1920x1080, FPS: 30
✓ camera2.mp4 - Resolution: 1920x1080, FPS: 30
✓ camera3.mp4 - Resolution: 1280x720, FPS: 25
✓ camera4.mp4 - Resolution: 1280x720, FPS: 30
✓ First frames displayed successfully
Phase 1: PASSED
```

---

## PHASE 2: Calibration Module

### Objectives
- Implement floor boundary point selection UI
- Implement grid layout arrangement UI
- Compute homography matrices
- Validate calibration data

### Background: Homography Explained
A homography is a 3x3 matrix that transforms points from one plane to another. For our use case:
- **Source plane**: Camera view (pixels in camera frame)
- **Destination plane**: Top-down floor map (pixels on canvas)
- **Transformation**: Maps where person appears in camera → where they are on floor map

OpenCV's `cv2.getPerspectiveTransform()` or `cv2.findHomography()` computes this matrix from 4 point correspondences.

### Tasks

#### Task 2.1: Implement Point Selection UI (calibration.py)

**What to implement:**

**Class: PointSelector**
```python
class PointSelector:
    def __init__(self, frame, camera_id):
        """Initialize with camera frame and ID"""
        
    def select_points(self):
        """
        Display frame, allow user to click 4 points
        Returns: list of 4 (x, y) tuples
        """
        # Display frame in window
        # Set mouse callback to capture clicks
        # Draw red circles at clicked points
        # Show instructions: "Click 4 corners of floor area. Press 'r' to reset, 'c' to confirm"
        # Wait for 4 points and confirmation
        # Return points
```

**Mouse Callback Logic:**
- Left click: Add point (up to 4)
- Right click or 'r' key: Reset all points
- 'c' key: Confirm (only if 4 points selected)
- 'q' key: Quit/cancel

**Visual Feedback:**
- Draw red circles at clicked points
- Draw lines connecting points in order
- Show count: "Points: 3/4"

**Validation:**
- Must have exactly 4 points
- Points should not be collinear (form valid quadrilateral)
- Points should be within frame bounds

**Rationale:**
- Manual point selection gives user full control
- Visual feedback ensures accuracy
- Reset option allows correction of mistakes

#### Task 2.2: Implement Grid Layout UI (calibration.py)

**What to implement:**

**Class: GridLayoutManager**
```python
class GridLayoutManager:
    def __init__(self, num_cameras, canvas_width, canvas_height):
        """Calculate grid dimensions and cell size"""
        
    def arrange_cameras(self, camera_frames):
        """
        Display grid canvas with camera thumbnails
        Allow user to drag/arrange cameras into grid cells
        Returns: dict mapping camera_id -> destination rectangle (4 corners)
        """
```

**UI Design:**
- Display blank canvas with grid lines
- Show camera thumbnails on the side or initially placed in default grid positions
- User can drag thumbnails to different grid cells
- Each cell shows which camera occupies it
- Instructions: "Arrange cameras to match room layout. Press 'c' to confirm"

**Alternative Simpler Approach (Recommended for AI Agent):**
- **Auto-arrangement**: Automatically place cameras in grid order (camera 0 → cell 0,0; camera 1 → cell 0,1, etc.)
- Show preview with labels
- User just confirms or manually adjusts if needed
- This is simpler to implement and test

**Destination Points Calculation:**
For camera in grid cell (row, col):
```
cell_width = canvas_width / num_cols
cell_height = canvas_height / num_rows

top_left = (col * cell_width, row * cell_height)
top_right = ((col + 1) * cell_width, row * cell_height)
bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
bottom_left = (col * cell_width, (row + 1) * cell_height)

destination_points = [top_left, top_right, bottom_right, bottom_left]
```

**Rationale:**
- Grid provides spatial context
- Auto-arrangement reduces user effort
- Clear visual representation of camera coverage

#### Task 2.3: Compute Homography Matrices (calibration.py)

**What to implement:**

**Function: compute_homography**
```python
def compute_homography(source_points, destination_points):
    """
    Compute homography matrix from source to destination points
    
    Args:
        source_points: 4 points in camera frame (x, y)
        destination_points: 4 points on canvas (x, y)
        
    Returns:
        homography_matrix: 3x3 numpy array
    """
    # Convert to numpy arrays
    # Use cv2.getPerspectiveTransform(src, dst)
    # Validate matrix is not singular
    # Return matrix
```

**Mathematics:**
- Source points: corners of floor area in camera view
- Destination points: corners of camera's grid cell on canvas
- OpenCV solves for H where: destination = H @ source
- Matrix is 3x3, allows for perspective transformation

**Validation:**
- Check matrix is not None
- Check determinant is non-zero (invertible matrix)
- Optionally test transform a few points to verify reasonableness

**Rationale:**
- Homography handles perspective distortion
- 4 points sufficient for unique solution
- OpenCV provides robust implementation

#### Task 2.4: Main Calibration Function (calibration.py)

**What to implement:**

**Function: calibrate_cameras**
```python
def calibrate_cameras(video_files, canvas_width, canvas_height):
    """
    Main calibration workflow
    
    Returns:
        calibration_data: dict with structure:
        {
            'canvas_width': int,
            'canvas_height': int,
            'cameras': [
                {
                    'camera_id': int,
                    'video_file': str,
                    'source_points': [(x,y), ...],  # 4 points
                    'destination_points': [(x,y), ...],  # 4 points
                    'homography_matrix': np.array,  # 3x3
                    'grid_cell': (row, col)
                },
                ...
            ]
        }
    """
    # Step 1: Load first frame from each video
    # Step 2: For each camera, run point selection UI
    # Step 3: Calculate grid layout
    # Step 4: Assign destination points based on grid
    # Step 5: Compute homography for each camera
    # Step 6: Return calibration data structure
```

**Error Handling:**
- If video fails to load: raise error with filename
- If user cancels point selection: raise error
- If homography computation fails: raise error with camera ID

### Testing Phase 2

#### Test Script: test_calibration.py

**What to test:**

**Test 2.1: Point Selection**
1. Load a single test frame
2. Run PointSelector
3. User clicks 4 points (manually test)
4. Print collected points
5. Verify 4 points collected, coordinates within frame

**Test 2.2: Grid Layout**
1. Create mock camera data (2-4 cameras)
2. Run GridLayoutManager
3. Print destination points for each camera
4. Verify each camera has 4 destination points
5. Verify points define rectangles within canvas bounds
6. Verify no overlapping rectangles (if strict grid)

**Test 2.3: Homography Computation**
1. Use sample source points: [(100, 100), (500, 100), (500, 400), (100, 400)]
2. Use sample destination points: [(0, 0), (600, 0), (600, 600), (0, 600)]
3. Compute homography
4. Print matrix
5. Test transform sample point (300, 250) → should be near (300, 300)
6. Verify transformed points within expected range

**Test 2.4: Full Calibration Workflow**
1. Run complete calibration with test videos
2. Complete all UI steps (manual)
3. Print final calibration data structure
4. Verify all cameras have valid homographies
5. Save calibration to JSON for inspection (optional)

**Success Criteria:**
- Point selection UI works, collects 4 valid points
- Grid layout generates correct destination rectangles
- Homography matrices computed successfully (3x3, non-singular)
- Full calibration completes without errors
- Manual test transforms produce sensible coordinates

**How to test:**
```bash
python tests/test_calibration.py
```

**Expected output:**
```
Testing Point Selection...
✓ Displayed frame for camera 0
✓ User selected 4 points: [(120, 150), (880, 160), (900, 680), (100, 670)]
✓ Points validated

Testing Grid Layout...
✓ Grid dimensions: 2x2 for 4 cameras
✓ Camera 0 -> cell (0,0) -> dest: [(0,0), (600,0), (600,600), (0,600)]
✓ Camera 1 -> cell (0,1) -> dest: [(600,0), (1200,0), (1200,600), (600,600)]
✓ All destination rectangles valid

Testing Homography...
✓ Homography matrix computed:
  [[ 1.2  0.1  -50]
   [ 0.05 1.3  -30]
   [ 0.0001 0.0002 1]]
✓ Test transform: (300, 250) -> (315, 295)
✓ Matrix is invertible

Testing Full Calibration...
✓ Loaded 4 video files
✓ Point selection completed for all cameras
✓ Grid layout arranged
✓ All homographies computed
✓ Calibration data structure valid

Phase 2: PASSED
```

---

## PHASE 3: Detection Module

### Objectives
- Load YOLOv8 model
- Detect persons in video frames
- Extract bounding boxes
- Calculate foot positions

### Background: Person Detection
YOLOv8 (You Only Look Once v8) is a real-time object detection model that:
- Processes entire image in one pass
- Returns bounding boxes, class labels, and confidence scores
- Class ID for "person" is typically 0 in COCO dataset
- Returns boxes in format: [x1, y1, x2, y2] (top-left, bottom-right)

### Tasks

#### Task 3.1: Initialize YOLO Model (detector.py)

**What to implement:**

**Class: PersonDetector**
```python
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_name: YOLO model variant (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detection
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # COCO dataset person class
```

**Model Variants:**
- yolov8n.pt: Nano (fastest, least accurate)
- yolov8s.pt: Small
- yolov8m.pt: Medium
- yolov8l.pt: Large
- yolov8x.pt: Extra large (slowest, most accurate)

**Recommendation:** Start with yolov8n for speed, upgrade if accuracy insufficient

**Rationale:**
- Ultralytics library handles model download automatically
- Pre-trained on COCO dataset (includes person class)
- Single-line initialization

#### Task 3.2: Implement Detection (detector.py)

**What to implement:**

**Method: detect_persons**
```python
def detect_persons(self, frame):
    """
    Detect persons in frame and return foot positions
    
    Args:
        frame: numpy array (BGR image)
        
    Returns:
        detections: list of dict with structure:
        [
            {
                'bbox': [x1, y1, x2, y2],  # Bounding box
                'confidence': float,
                'foot_position': (x, y)  # Bottom-center of bbox
            },
            ...
        ]
    """
    # Run YOLO inference
    results = self.model(frame, verbose=False)
    
    # Filter for person class and confidence threshold
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == self.person_class_id and box.conf >= self.confidence_threshold:
                # Extract bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate foot position (bottom-center)
                foot_x = (x1 + x2) / 2
                foot_y = y2  # Bottom edge
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf),
                    'foot_position': (foot_x, foot_y)
                })
    
    return detections
```

**Foot Position Logic:**
- Assumption: Person's feet are at bottom of bounding box
- X-coordinate: Center of box (average of left and right)
- Y-coordinate: Bottom edge of box
- This represents where person is standing on the floor

**Rationale:**
- Foot position is what we need to map to floor plan
- Bottom-center is best estimate of floor contact point
- More accurate than using center of bounding box

#### Task 3.3: Batch Processing (detector.py)

**What to implement:**

**Method: detect_persons_batch**
```python
def detect_persons_batch(self, frames):
    """
    Detect persons in multiple frames (one per camera)
    
    Args:
        frames: list of numpy arrays
        
    Returns:
        all_detections: list of detection lists (one per frame)
    """
    # Process all frames
    # Can use YOLO batch inference for efficiency
    # Return list of detection lists
```

**Rationale:**
- Process all camera frames in one call
- YOLO can batch process for GPU efficiency
- Simplifies main loop

### Testing Phase 3

#### Test Script: test_detector.py

**What to test:**

**Test 3.1: Model Loading**
1. Initialize PersonDetector
2. Verify model loaded successfully
3. Print model info

**Test 3.2: Single Frame Detection**
1. Load a test frame with visible person
2. Run detection
3. Print number of detections
4. Print bounding boxes and foot positions
5. Draw bounding boxes and foot dots on frame
6. Display annotated frame
7. Verify detections visually

**Test 3.3: Detection Accuracy**
1. Test frame with 0 persons → expect 0 detections
2. Test frame with 1 person → expect 1 detection
3. Test frame with multiple persons → expect multiple detections
4. Test with different confidence thresholds

**Test 3.4: Batch Processing**
1. Load frames from multiple cameras
2. Run batch detection
3. Print detection counts per camera
4. Verify results consistent with single-frame processing

**Success Criteria:**
- Model loads without errors
- Detections found in frames with people
- Foot positions at bottom-center of bounding boxes
- No false positives on empty frames (or acceptable rate)
- Batch processing produces same results as sequential

**How to test:**
```bash
python tests/test_detector.py
```

**Expected output:**
```
Testing Model Loading...
✓ YOLOv8n model loaded successfully
✓ Model info: yolov8n.pt, 3.2M parameters

Testing Single Frame Detection...
✓ Loaded test frame: 1920x1080
✓ Detection completed in 45ms
✓ Found 2 persons
  Person 1: bbox=[450, 120, 680, 890], conf=0.89, foot=(565, 890)
  Person 2: bbox=[1100, 200, 1250, 780], conf=0.76, foot=(1175, 780)
✓ Annotated frame displayed

Testing Detection Accuracy...
✓ Empty frame: 0 detections (correct)
✓ Single person frame: 1 detection (correct)
✓ Multiple person frame: 3 detections (correct)

Testing Batch Processing...
✓ Batch detection on 4 frames
✓ Camera 0: 2 persons, Camera 1: 1 person, Camera 2: 0 persons, Camera 3: 3 persons
✓ Results match sequential processing

Phase 3: PASSED
```

---

## PHASE 4: Transformation Module

### Objectives
- Transform foot positions from camera coordinates to canvas coordinates
- Apply homography transformations
- Validate transformed coordinates

### Background: Applying Homography
Given homography matrix H and point (x, y) in camera frame:
1. Convert to homogeneous coordinates: [x, y, 1]
2. Apply transformation: [x', y', w'] = H @ [x, y, 1]
3. Convert back: (x'/w', y'/w')

OpenCV provides `cv2.perspectiveTransform()` for this.

### Tasks

#### Task 4.1: Implement Transformation (transformer.py)

**What to implement:**

**Function: transform_point**
```python
import cv2
import numpy as np

def transform_point(point, homography_matrix):
    """
    Transform single point using homography
    
    Args:
        point: (x, y) tuple in source coordinates
        homography_matrix: 3x3 numpy array
        
    Returns:
        transformed_point: (x, y) tuple in destination coordinates
    """
    # Convert point to correct format for cv2.perspectiveTransform
    point_array = np.array([[point]], dtype=np.float32)
    
    # Apply transformation
    transformed = cv2.perspectiveTransform(point_array, homography_matrix)
    
    # Extract result
    x, y = transformed[0][0]
    
    return (float(x), float(y))
```

**Function: transform_points**
```python
def transform_points(points, homography_matrix):
    """
    Transform multiple points using homography
    
    Args:
        points: list of (x, y) tuples
        homography_matrix: 3x3 numpy array
        
    Returns:
        transformed_points: list of (x, y) tuples
    """
    if len(points) == 0:
        return []
    
    # Convert to numpy array format
    points_array = np.array([points], dtype=np.float32)
    
    # Apply transformation
    transformed = cv2.perspectiveTransform(points_array, homography_matrix)
    
    # Convert back to list of tuples
    return [(float(x), float(y)) for x, y in transformed[0]]
```

**Rationale:**
- OpenCV handles homography math internally
- Batch transformation more efficient than looping
- Type conversions ensure compatibility

#### Task 4.2: Coordinate Validation (transformer.py)

**What to implement:**

**Function: validate_canvas_coordinates**
```python
def validate_canvas_coordinates(points, canvas_width, canvas_height):
    """
    Check if points are within canvas bounds
    
    Args:
        points: list of (x, y) tuples
        canvas_width: int
        canvas_height: int
        
    Returns:
        valid_points: list of (x, y) tuples within bounds
        invalid_count: number of points outside bounds
    """
    valid_points = []
    invalid_count = 0
    
    for x, y in points:
        if 0 <= x <= canvas_width and 0 <= y <= canvas_height:
            valid_points.append((x, y))
        else:
            invalid_count += 1
            # Optionally log warning
    
    return valid_points, invalid_count
```

**Rationale:**
- Detections outside camera's calibrated floor area will map outside canvas
- Filter out invalid points to avoid drawing errors
- Track invalid count for debugging

#### Task 4.3: Multi-Camera Transformation (transformer.py)

**What to implement:**

**Class: CoordinateTransformer**
```python
class CoordinateTransformer:
    def __init__(self, calibration_data):
        """
        Initialize with calibration data
        
        Args:
            calibration_data: dict from calibration module
        """
        self.canvas_width = calibration_data['canvas_width']
        self.canvas_height = calibration_data['canvas_height']
        self.camera_homographies = {}
        
        for camera in calibration_data['cameras']:
            camera_id = camera['camera_id']
            self.camera_homographies[camera_id] = camera['homography_matrix']
    
    def transform_detections(self, detections_per_camera):
        """
        Transform detections from all cameras to canvas coordinates
        
        Args:
            detections_per_camera: dict mapping camera_id -> list of detections
            
        Returns:
            canvas_points: list of (x, y) tuples on canvas
        """
        canvas_points = []
        
        for camera_id, detections in detections_per_camera.items():
            if camera_id not in self.camera_homographies:
                continue
                
            homography = self.camera_homographies[camera_id]
            
            # Extract foot positions
            foot_positions = [d['foot_position'] for d in detections]
            
            # Transform to canvas coordinates
            if foot_positions:
                transformed = transform_points(foot_positions, homography)
                
                # Validate and add to canvas points
                valid_points, invalid_count = validate_canvas_coordinates(
                    transformed, self.canvas_width, self.canvas_height
                )
                canvas_points.extend(valid_points)
        
        return canvas_points
```

**Rationale:**
- Encapsulates transformation logic
- Handles multiple cameras uniformly
- Aggregates all detections into single list for visualization

### Testing Phase 4

#### Test Script: test_transformer.py

**What to test:**

**Test 4.1: Single Point Transformation**
1. Create test homography (identity or simple scaling)
2. Transform test point
3. Verify result matches expected
4. Test with actual calibration homography from Phase 2

**Test 4.2: Batch Transformation**
1. Create list of test points
2. Transform batch
3. Verify all points transformed correctly
4. Compare with individual transforms (should match)

**Test 4.3: Coordinate Validation**
1. Create points inside and outside canvas bounds
2. Run validation
3. Verify correct filtering
4. Check invalid count accurate

**Test 4.4: Multi-Camera Transformation**
1. Create mock calibration data with 2 cameras
2. Create mock detections for each camera
3. Run CoordinateTransformer
4. Verify all detections transformed
5. Verify points within canvas bounds
6. Print transformed coordinates

**Success Criteria:**
- Transformations produce expected coordinates
- Batch and individual transforms consistent
- Validation correctly filters out-of-bounds points
- Multi-camera transformation aggregates all detections correctly

**How to test:**
```bash
python tests/test_transformer.py
```

**Expected output:**
```
Testing Single Point Transformation...
✓ Identity transform: (100, 200) -> (100, 200)
✓ Scaling transform: (100, 200) -> (200, 400)
✓ Calibration transform: (565, 890) -> (315, 580)

Testing Batch Transformation...
✓ Transformed 5 points
✓ Results match individual transforms

Testing Coordinate Validation...
✓ Valid points: [(100, 200), (500, 600)]
✓ Invalid count: 2
✓ Out-of-bounds points filtered correctly

Testing Multi-Camera Transformation...
✓ Initialized transformer with 2 cameras
✓ Camera 0: 3 detections -> 3 canvas points
✓ Camera 1: 2 detections -> 2 canvas points
✓ Total canvas points: 5
✓ All points within bounds
  Points: [(315, 580), (420, 650), (180, 490), (850, 420), (920, 510)]

Phase 4: PASSED
```

---

## PHASE 5: Visualization Module

### Objectives
- Create 2D map canvas
- Draw detection points
- Update display in real-time
- Handle user input (pause, quit)

### Tasks

#### Task 5.1: Canvas Initialization (visualizer.py)

**What to implement:**

**Class: MapVisualizer**
```python
import cv2
import numpy as np

class MapVisualizer:
    def __init__(self, canvas_width, canvas_height, background_color=(50, 50, 50)):
        """
        Initialize map visualizer
        
        Args:
            canvas_width: int
            canvas_height: int
            background_color: (B, G, R) tuple
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.background_color = background_color
        self.window_name = "2D Event Map"
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, canvas_width, canvas_height)
```

**Rationale:**
- Named window allows resizing
- Fixed background color for consistency
- Separate window for map display

#### Task 5.2: Drawing Functions (visualizer.py)

**What to implement:**

**Method: create_blank_canvas**
```python
def create_blank_canvas(self):
    """
    Create blank canvas with background color
    
    Returns:
        canvas: numpy array (BGR image)
    """
    canvas = np.full(
        (self.canvas_height, self.canvas_width, 3),
        self.background_color,
        dtype=np.uint8
    )
    return canvas
```

**Method: draw_grid** (optional, helpful for debugging)
```python
def draw_grid(self, canvas, rows, cols, color=(80, 80, 80)):
    """
    Draw grid lines on canvas
    
    Args:
        canvas: numpy array to draw on
        rows: number of grid rows
        cols: number of grid columns
        color: (B, G, R) tuple
    """
    cell_width = self.canvas_width // cols
    cell_height = self.canvas_height // rows
    
    # Vertical lines
    for i in range(1, cols):
        x = i * cell_width
        cv2.line(canvas, (x, 0), (x, self.canvas_height), color, 1)
    
    # Horizontal lines
    for i in range(1, rows):
        y = i * cell_height
        cv2.line(canvas, (0, y), (self.canvas_width, y), color, 1)
```

**Method: draw_detections**
```python
def draw_detections(self, canvas, points, color=(0, 255, 0), radius=8, thickness=-1):
    """
    Draw detection points on canvas
    
    Args:
        canvas: numpy array to draw on
        points: list of (x, y) tuples
        color: (B, G, R) tuple
        radius: circle radius
        thickness: -1 for filled circle
    """
    for x, y in points:
        # Convert to int for drawing
        center = (int(x), int(y))
        cv2.circle(canvas, center, radius, color, thickness)
```

**Rationale:**
- Separate drawing functions for modularity
- Grid helps visualize camera regions during testing
- Circles are simple and visible

#### Task 5.3: Display and Update (visualizer.py)

**What to implement:**

**Method: show**
```python
def show(self, canvas, wait_key=1):
    """
    Display canvas in window
    
    Args:
        canvas: numpy array to display
        wait_key: milliseconds to wait (1 for real-time, 0 for wait until keypress)
        
    Returns:
        key: pressed key code (-1 if no key pressed)
    """
    cv2.imshow(self.window_name, canvas)
    key = cv2.waitKey(wait_key)
    return key
```

**Method: render_frame**
```python
def render_frame(self, detection_points, show_grid=False, grid_rows=2, grid_cols=2):
    """
    Render complete frame with detections
    
    Args:
        detection_points: list of (x, y) tuples
        show_grid: bool, whether to draw grid
        grid_rows, grid_cols: grid dimensions
        
    Returns:
        canvas: rendered canvas
    """
    # Create blank canvas
    canvas = self.create_blank_canvas()
    
    # Optionally draw grid
    if show_grid:
        self.draw_grid(canvas, grid_rows, grid_cols)
    
    # Draw detections
    self.draw_detections(canvas, detection_points)
    
    return canvas
```

**Method: close**
```python
def close(self):
    """Close visualization window"""
    cv2.destroyWindow(self.window_name)
```

**Rationale:**
- render_frame creates complete visualization
- Separate show method for display control
- Clean shutdown with close method

### Testing Phase 5

#### Test Script: test_visualizer.py

**What to test:**

**Test 5.1: Canvas Creation**
1. Initialize MapVisualizer
2. Create blank canvas
3. Display canvas
4. Verify correct size and color

**Test 5.2: Grid Drawing**
1. Create canvas
2. Draw 2x2 grid
3. Display
4. Visually verify grid lines correct

**Test 5.3: Point Drawing**
1. Create canvas
2. Draw test points at various positions:
   - Center: (600, 600)
   - Corners: (50, 50), (1150, 50), (1150, 1150), (50, 1150)
   - Random: (300, 400), (800, 700)
3. Display
4. Visually verify dots appear at correct locations

**Test 5.4: Animation Test**
1. Create moving point simulation
2. Loop: update point position, render, display
3. Verify smooth animation
4. Test keyboard input (press 'q' to quit)

**Test 5.5: Full Rendering**
1. Create multiple test points (simulating detections)
2. Render complete frame with grid and points
3. Display
4. Verify all elements visible

**Success Criteria:**
- Canvas displays correctly
- Grid lines visible and aligned
- Points draw at correct coordinates
- Animation runs smoothly
- Keyboard input works
- No display errors or crashes

**How to test:**
```bash
python tests/test_visualizer.py
```

**Expected output:**
```
Testing Canvas Creation...
✓ MapVisualizer initialized (1200x1200)
✓ Blank canvas created
✓ Canvas displayed

Testing Grid Drawing...
✓ 2x2 grid drawn
✓ Grid lines visible at correct positions

Testing Point Drawing...
✓ Drew 7 test points
✓ All points visible at correct locations
✓ Circle size and color correct

Testing Animation...
✓ Running 100 frame animation
✓ Smooth rendering at ~30 FPS
✓ Keyboard input responsive ('q' to quit)

Testing Full Rendering...
✓ Rendered frame with 12 detection points
✓ Grid overlay visible
✓ All points correctly positioned

Phase 5: PASSED
```

---

## PHASE 6: Main Orchestrator

### Objectives
- Integrate all modules
- Implement video processing loop
- Synchronize multi-camera frames
- Handle user controls (play, pause, quit)

### Tasks

#### Task 6.1: Video Loading (main.py)

**What to implement:**

**Function: load_videos**
```python
def load_videos(video_files):
    """
    Load video captures for all cameras
    
    Args:
        video_files: list of video file paths
        
    Returns:
        captures: dict mapping camera_id -> cv2.VideoCapture
        metadata: dict with video info (fps, frame_count, etc.)
    """
    captures = {}
    metadata = {
        'fps': [],
        'frame_counts': [],
        'resolutions': []
    }
    
    for camera_id, video_file in enumerate(video_files):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_file}")
        
        captures[camera_id] = cap
        
        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata['fps'].append(fps)
        metadata['frame_counts'].append(frame_count)
        metadata['resolutions'].append((width, height))
        
        print(f"Camera {camera_id}: {video_file}")
        print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
    
    return captures, metadata
```

**Rationale:**
- Centralized video loading with error handling
- Collect metadata for debugging and sync
- Dictionary structure for easy camera ID lookup

#### Task 6.2: Frame Synchronization (main.py)

**What to implement:**

**Function: read_synchronized_frames**
```python
def read_synchronized_frames(captures):
    """
    Read one frame from each active camera
    
    Args:
        captures: dict mapping camera_id -> cv2.VideoCapture
        
    Returns:
        frames: dict mapping camera_id -> frame (only active cameras)
        all_finished: bool, True if all cameras finished
    """
    frames = {}
    active_count = 0
    
    for camera_id, cap in captures.items():
        ret, frame = cap.read()
        if ret:
            frames[camera_id] = frame
            active_count += 1
    
    all_finished = (active_count == 0)
    
    return frames, all_finished
```

**Rationale:**
- Reads one frame from each camera per call
- Automatically handles cameras that finish early
- Returns only active frames (ignores finished cameras)
- Clear termination condition

#### Task 6.3: Main Processing Loop (main.py)

**What to implement:**

**Function: main**
```python
def main():
    """Main execution function"""
    
    # Import config
    from config import (
        VIDEO_FILES, CANVAS_WIDTH, CANVAS_HEIGHT,
        DETECTION_CONFIDENCE_THRESHOLD, YOLO_MODEL
    )
    
    print("=== 2D Event Map System ===\n")
    
    # Phase 1: Load videos
    print("Loading videos...")
    captures, metadata = load_videos(VIDEO_FILES)
    num_cameras = len(VIDEO_FILES)
    print(f"\n✓ Loaded {num_cameras} cameras\n")
    
    # Phase 2: Calibration
    print("Starting calibration...")
    print("Instructions:")
    print("  1. Click 4 corners of floor area in each camera view")
    print("  2. Press 'c' to confirm, 'r' to reset")
    print("  3. Arrange cameras on grid (auto-arranged by default)")
    print()
    
    from calibration import calibrate_cameras
    calibration_data = calibrate_cameras(VIDEO_FILES, CANVAS_WIDTH, CANVAS_HEIGHT)
    print("\n✓ Calibration complete\n")
    
    # Reset video captures to start
    for cap in captures.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Phase 3: Initialize detector
    print("Loading person detector...")
    from detector import PersonDetector
    detector = PersonDetector(
        model_name=YOLO_MODEL,
        confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD
    )
    print("✓ Detector loaded\n")
    
    # Phase 4: Initialize transformer
    print("Initializing coordinate transformer...")
    from transformer import CoordinateTransformer
    transformer = CoordinateTransformer(calibration_data)
    print("✓ Transformer ready\n")
    
    # Phase 5: Initialize visualizer
    print("Setting up visualization...")
    from visualizer import MapVisualizer
    visualizer = MapVisualizer(CANVAS_WIDTH, CANVAS_HEIGHT)
    
    # Calculate grid dimensions for display
    from utils import calculate_grid_dimensions
    grid_rows, grid_cols = calculate_grid_dimensions(num_cameras)
    print("✓ Visualizer ready\n")
    
    # Phase 6: Main processing loop
    print("Starting detection and tracking...")
    print("Controls: 'q' to quit, 'p' to pause, SPACE to resume\n")
    
    frame_count = 0
    paused = False
    
    while True:
        # Handle pause
        if paused:
            key = cv2.waitKey(10) & 0xFF
            if key == ord(' '):
                paused = False
                print("Resumed")
            elif key == ord('q'):
                break
            continue
        
        # Read frames from all cameras
        frames, all_finished = read_synchronized_frames(captures)
        
        if all_finished:
            print("\nAll videos finished")
            break
        
        # Detect persons in all frames
        detections_per_camera = {}
        for camera_id, frame in frames.items():
            detections = detector.detect_persons(frame)
            detections_per_camera[camera_id] = detections
        
        # Transform to canvas coordinates
        canvas_points = transformer.transform_detections(detections_per_camera)
        
        # Render and display
        canvas = visualizer.render_frame(
            canvas_points,
            show_grid=True,
            grid_rows=grid_rows,
            grid_cols=grid_cols
        )
        
        key = visualizer.show(canvas, wait_key=1)
        
        # Handle keyboard input
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('p'):
            paused = True
            print("Paused (press SPACE to resume)")
        
        frame_count += 1
        if frame_count % 30 == 0:  # Progress update every 30 frames
            total_detections = sum(len(d) for d in detections_per_camera.values())
            print(f"Frame {frame_count}: {len(frames)} cameras active, {total_detections} persons detected")
    
    # Cleanup
    print("\nCleaning up...")
    for cap in captures.values():
        cap.release()
    visualizer.close()
    cv2.destroyAllWindows()
    print("✓ Done")

if __name__ == "__main__":
    main()
```

**Rationale:**
- Clear phase-by-phase execution
- Progress messages for user feedback
- Pause/resume functionality for debugging
- Graceful handling of video finish
- Proper cleanup on exit

#### Task 6.4: Error Handling (main.py)

**What to add:**

Wrap main execution in try-except:
```python
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
```

**Rationale:**
- Clean exit on Ctrl+C
- Error messages for debugging
- Always close windows

### Testing Phase 6

#### Test Script: Integration Testing

**Test 6.1: Dry Run with Minimal Frames**
1. Modify main to process only first 10 frames
2. Run complete pipeline
3. Verify all phases execute
4. Check for errors

**Test 6.2: Single Camera Test**
1. Configure with only 1 video file
2. Run complete pipeline
3. Verify works with single camera

**Test 6.3: Multi-Camera Test**
1. Configure with 2-4 video files
2. Run complete pipeline
3. Verify all cameras processed
4. Check detections from all cameras appear on map

**Test 6.4: Different Length Videos**
1. Use videos of different durations
2. Run pipeline
3. Verify shorter videos finish gracefully
4. Verify longer videos continue processing

**Test 6.5: User Controls**
1. Run pipeline
2. Test pause ('p') and resume (SPACE)
3. Test quit ('q')
4. Verify responsive controls

**Test 6.6: Performance Test**
1. Run on longer videos (100+ frames)
2. Measure FPS
3. Check for memory leaks (monitor RAM usage)
4. Verify stable performance

**Success Criteria:**
- Complete pipeline executes without errors
- All cameras processed correctly
- Detections appear on map in correct positions
- Videos of different lengths handled gracefully
- User controls responsive
- Performance acceptable (>10 FPS on reasonable hardware)
- No crashes or memory issues

**How to test:**
```bash
python main.py
```

**Expected output:**
```
=== 2D Event Map System ===

Loading videos...
Camera 0: /path/to/camera1.mp4
  Resolution: 1920x1080, FPS: 30.0, Frames: 450
Camera 1: /path/to/camera2.mp4
  Resolution: 1920x1080, FPS: 30.0, Frames: 380
Camera 2: /path/to/camera3.mp4
  Resolution: 1280x720, FPS: 25.0, Frames: 420

✓ Loaded 3 cameras

Starting calibration...
Instructions:
  1. Click 4 corners of floor area in each camera view
  2. Press 'c' to confirm, 'r' to reset
  3. Arrange cameras on grid (auto-arranged by default)

[User completes calibration]

✓ Calibration complete

Loading person detector...
✓ Detector loaded

Initializing coordinate transformer...
✓ Transformer ready

Setting up visualization...
✓ Visualizer ready

Starting detection and tracking...
Controls: 'q' to quit, 'p' to pause, SPACE to resume

Frame 30: 3 cameras active, 4 persons detected
Frame 60: 3 cameras active, 5 persons detected
Frame 90: 3 cameras active, 3 persons detected
...
Frame 380: 2 cameras active, 2 persons detected
Frame 390: 1 cameras active, 1 persons detected
Frame 420: 1 cameras active, 0 persons detected

All videos finished

Cleaning up...
✓ Done
```

---

## PHASE 7: Error Handling & Validation

### Objectives
- Add comprehensive error handling
- Validate inputs at each stage
- Provide helpful error messages
- Handle edge cases gracefully

### Tasks

#### Task 7.1: Input Validation (utils.py)

**What to implement:**

**Enhanced validate_video_files:**
```python
def validate_video_files(video_paths):
    """
    Validate video files exist and are readable
    
    Returns:
        True if all valid, raises ValueError otherwise
    """
    if not video_paths:
        raise ValueError("No video files specified in config")
    
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        
        # Try to open
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        cap.release()
    
    return True
```

**Add validation helpers:**
```python
def validate_point_count(points, expected=4):
    """Validate correct number of points"""
    if len(points) != expected:
        raise ValueError(f"Expected {expected} points, got {len(points)}")

def validate_points_not_collinear(points):
    """Check points form valid quadrilateral"""
    # Check not all on same line
    # Simple check: area of polygon > threshold
    import cv2
    area = cv2.contourArea(np.array(points, dtype=np.float32))
    if area < 100:  # Arbitrary threshold
        raise ValueError("Points are too close or collinear")

def validate_homography_matrix(matrix):
    """Validate homography matrix is valid"""
    if matrix is None:
        raise ValueError("Homography computation failed")
    
    # Check not singular
    det = np.linalg.det(matrix)
    if abs(det) < 1e-6:
        raise ValueError("Homography matrix is singular (determinant near zero)")
```

#### Task 7.2: Error Handling in Modules

**Add to calibration.py:**
```python
# In PointSelector.select_points():
try:
    # ... point selection logic ...
    validate_point_count(self.points, 4)
    validate_points_not_collinear(self.points)
    return self.points
except Exception as e:
    raise RuntimeError(f"Point selection failed for camera {self.camera_id}: {e}")

# In compute_homography():
try:
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    validate_homography_matrix(H)
    return H
except Exception as e:
    raise RuntimeError(f"Homography computation failed: {e}")
```

**Add to detector.py:**
```python
# In PersonDetector.__init__():
try:
    self.model = YOLO(model_name)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model {model_name}: {e}")

# In detect_persons():
try:
    results = self.model(frame, verbose=False)
    # ... detection logic ...
except Exception as e:
    # Log error but don't crash - return empty detections
    print(f"Warning: Detection failed: {e}")
    return []
```

**Add to main.py:**
```python
# Before calibration:
from utils import validate_video_files
try:
    validate_video_files(VIDEO_FILES)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
```

#### Task 7.3: Edge Case Handling

**Handle empty detections:**
- If no persons detected in any camera: display empty map (no crashes)

**Handle calibration cancellation:**
- If user closes window during calibration: exit gracefully with message

**Handle video read errors:**
- If frame read fails mid-processing: skip frame, continue with others

**Handle keyboard interrupt:**
- Ctrl+C during any phase: cleanup and exit cleanly

### Testing Phase 7

**Test 7.1: Invalid Video Path**
1. Set non-existent path in config
2. Run main
3. Expect clear error message

**Test 7.2: Corrupted Video**
1. Use corrupted/unreadable video file
2. Run main
3. Expect error during loading

**Test 7.3: Invalid Calibration**
1. During calibration, click only 3 points
2. Try to confirm
3. Expect validation error

**Test 7.4: Collinear Points**
1. During calibration, click 4 points in straight line
2. Try to confirm
3. Expect validation error

**Test 7.5: Empty Frame**
1. Use video with no people
2. Run complete pipeline
3. Verify no crashes, empty map displayed

**Test 7.6: Keyboard Interrupt**
1. Start main
2. Press Ctrl+C during various phases
3. Verify cleanup happens

**Success Criteria:**
- All error cases handled gracefully
- Clear error messages guide user
- No unhandled exceptions
- Proper cleanup on all exit paths

---

## PHASE 8: Optimization & Polish

### Objectives
- Improve performance
- Add useful features
- Refine user experience

### Optional Enhancements

#### Enhancement 8.1: Performance
- GPU acceleration for YOLO (automatic with CUDA)
- Multi-threaded detection (one thread per camera)
- Frame skipping option (process every Nth frame)

#### Enhancement 8.2: Visualization
- Color-code dots by camera
- Show trajectory trails (last N positions)
- Display statistics (person count, FPS)
- Add timestamp overlay

#### Enhancement 8.3: Calibration
- Save/load calibration to JSON file
- Skip calibration if file exists
- Visual preview of camera coverage on grid

#### Enhancement 8.4: Configuration
- Command-line arguments for video paths
- Config file (YAML/JSON) instead of hardcoded
- Adjustable parameters without code changes

---

## Dependencies & Installation

### requirements.txt
```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
```

### Installation Steps
```bash
# Navigate to project directory
cd /home/ezio/Documents/work/vis-tracking

# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# First run will download YOLOv8 weights automatically
```

---

## Execution Workflow

### Initial Setup
1. Clone/create project structure
2. Install dependencies
3. Configure video file paths in config.py
4. Ensure test videos are accessible

### Running the System
```bash
python main.py
```

### Calibration Workflow
1. System loads first frame from each camera
2. For each camera:
   - Window displays frame
   - User clicks 4 floor boundary corners
   - User presses 'c' to confirm
3. Grid arrangement displayed (auto-arranged)
4. User confirms or adjusts
5. Homographies computed and stored

### Runtime Workflow
1. Videos reset to start
2. Detection model loaded
3. Main loop begins:
   - Read synchronized frames
   - Detect persons in each frame
   - Transform coordinates
   - Render and display map
   - Handle user input
4. Continue until videos finish or user quits
5. Cleanup and exit

---

## Expected Challenges & Solutions

### Challenge 1: Calibration Accuracy
**Problem:** User clicks imprecise points, homography inaccurate
**Solution:**
- Zoom feature for precise clicking
- Show preview of transformed region
- Allow re-calibration

### Challenge 2: Detection False Positives
**Problem:** YOLO detects non-persons or misses persons
**Solution:**
- Adjust confidence threshold
- Use larger YOLO model for better accuracy
- Add temporal filtering (track across frames)

### Challenge 3: Overlapping Camera Views
**Problem:** Same person detected by multiple cameras, appears as multiple dots
**Solution:**
- Implement spatial clustering (Phase 9)
- Merge detections within distance threshold
- Use tracking IDs for association

### Challenge 4: Performance
**Problem:** Processing too slow for real-time feel
**Solution:**
- Use YOLOv8n (fastest variant)
- Process every 2nd or 3rd frame
- Reduce video resolution
- Use GPU if available

### Challenge 5: Video Synchronization
**Problem:** Videos have different frame rates
**Solution:**
- Current approach: process frame-by-frame (assumes sync)
- Advanced: timestamp-based sync (requires metadata)
- Acceptable: slight temporal mismatch for recorded videos

---

## Testing Summary

### Per-Phase Testing
- Each phase has dedicated test script
- Tests can be run independently
- Validates functionality before proceeding

### Integration Testing
- Full pipeline test with main.py
- Various video configurations
- Edge cases and error conditions

### Manual Validation
- Visual inspection of calibration accuracy
- Detection accuracy on known frames
- Map positioning correctness

---

## Success Metrics

### Functional Requirements
✓ System loads multiple video files
✓ Calibration UI collects floor boundaries
✓ Homography transforms computed correctly
✓ Person detection identifies people
✓ Coordinates transform to map
✓ Map displays all detections
✓ Videos process synchronously
✓ User controls work (pause, quit)

### Quality Requirements
✓ No crashes or unhandled exceptions
✓ Clear error messages
✓ Acceptable performance (>10 FPS)
✓ Accurate positioning on map
✓ Clean code structure
✓ Modular and testable

---

## Future Enhancements (Post-Implementation)

### Phase 9: Person Tracking
- Assign unique IDs to persons
- Track individuals across frames and cameras
- Display trajectory history
- Count entries/exits

### Phase 10: Persistence
- Save calibration to file
- Load existing calibration
- Save detection events to database
- Export heatmaps

### Phase 11: Live Camera Support
- Replace video files with camera streams
- Real-time processing
- Network camera support (RTSP)

### Phase 12: Web Interface
- Web-based visualization
- Remote access
- Configuration interface
- Playback controls

---

## AI Agent Execution Guide

### Step-by-Step Process for AI Agent

**For each phase:**
1. Read phase objectives and tasks
2. Create/modify specified files
3. Implement functions/classes as described
4. Add error handling
5. Create test script
6. Run tests and verify success criteria
7. Document any issues or deviations
8. Proceed to next phase only after tests pass

**Critical Notes for AI Agent:**
- Follow implementation order strictly
- Test each phase before proceeding
- Use exact function/class names as specified
- Maintain consistent coding style
- Add comments for clarity
- Handle errors at every level
- Validate inputs before processing
- Clean up resources (video captures, windows)

**When encountering issues:**
- Check previous phase dependencies
- Verify imports and file paths
- Print debug information
- Test with simpler inputs first
- Consult error messages carefully

**Communication:**
- Report progress after each phase
- Describe any deviations from plan
- Ask for clarification if requirements ambiguous
- Summarize test results

---

## Conclusion

This implementation plan provides a comprehensive roadmap for building a multi-camera 2D event mapping system. The modular architecture, incremental testing approach, and detailed task breakdowns ensure systematic development and high code quality.

The plan is designed to be executed step-by-step by an AI agent, with clear objectives, rationale for decisions, and success criteria at each stage. Following this plan will result in a functional system that tracks people across multiple camera views and visualizes their positions on a unified 2D floor map.

**Ready to begin implementation? Start with Phase 1.**
