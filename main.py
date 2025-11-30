"""
Main entry point for multi-camera person tracking system.

This module coordinates the entire tracking workflow:
1. Load configuration and initialize components
2. Run calibration mode to set up camera-to-floor transformations
3. Run detection and tracking mode to process video feeds
4. Display results on 2D floor map in real-time
"""
