#!/usr/bin/env python3
"""
PageSnap - Automatic page turn detection for book scanning
Uses Continuity Camera (iPhone as webcam) with OpenCV frame differencing
"""

import cv2
import numpy as np
import time
import os
import platform
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

# Fix for macOS - must be before any cv2 window operations
if platform.system() == "Darwin":
    os.environ["QT_QPA_PLATFORM"] = "cocoa"


class State(Enum):
    IDLE = "Monitoring..."
    TURNING = "Page turning..."
    STABILIZING = "Stabilizing..."
    CAPTURING = "Captured!"
    COOLDOWN = "Cooldown..."


@dataclass
class Config:
    motion_threshold: float = 15.0
    stability_threshold: float = 5.0
    stability_delay: float = 1.0
    cooldown_period: float = 2.0
    blur_kernel: int = 21
    jpeg_quality: int = 90
    detection_scale: float = 0.25  # Scale down for detection processing


class ROISelector:
    """Handles ROI selection via mouse drag"""
    
    def __init__(self):
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                self.roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    
    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self.roi
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI rectangle and dim outside area"""
        display = frame.copy()
        
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)
        elif self.roi:
            x, y, w, h = self.roi
            # Dim outside ROI
            overlay = display.copy()
            overlay[:] = (overlay * 0.4).astype(np.uint8)
            overlay[y:y+h, x:x+w] = display[y:y+h, x:x+w]
            display = overlay
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return display


class PageSnapDetector:
    """Core page turn detection using frame differencing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = State.IDLE
        self.previous_frame: Optional[np.ndarray] = None
        self.stability_start: Optional[float] = None
        self.cooldown_start: Optional[float] = None
        self.last_delta: float = 0.0
    
    def process_frame(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[State, bool]:
        """
        Process a frame and return (current_state, should_capture)
        """
        should_capture = False
        
        # Convert and preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Crop to ROI if defined
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        # Scale down for faster processing
        small = cv2.resize(gray, None, fx=self.config.detection_scale, fy=self.config.detection_scale)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(small, (self.config.blur_kernel, self.config.blur_kernel), 0)
        
        if self.previous_frame is None:
            self.previous_frame = blurred
            return self.state, False
        
        # Calculate frame difference
        diff = cv2.absdiff(blurred, self.previous_frame)
        self.last_delta = np.mean(diff)
        
        now = time.time()
        
        # State machine
        if self.state == State.IDLE:
            if self.last_delta > self.config.motion_threshold:
                self.state = State.TURNING
        
        elif self.state == State.TURNING:
            if self.last_delta < self.config.stability_threshold:
                self.state = State.STABILIZING
                self.stability_start = now
        
        elif self.state == State.STABILIZING:
            if self.last_delta > self.config.stability_threshold:
                self.state = State.TURNING
            elif self.stability_start and (now - self.stability_start) >= self.config.stability_delay:
                self.state = State.CAPTURING
                should_capture = True
                self.cooldown_start = now
        
        elif self.state == State.CAPTURING:
            self.state = State.COOLDOWN
        
        elif self.state == State.COOLDOWN:
            if self.cooldown_start and (now - self.cooldown_start) >= self.config.cooldown_period:
                self.state = State.IDLE
        
        self.previous_frame = blurred
        return self.state, should_capture


class PageSnap:
    """Main application class"""
    
    def __init__(self, camera_index: int = 0, session_name: Optional[str] = None):
        self.config = Config()
        self.detector = PageSnapDetector(self.config)
        self.roi_selector = ROISelector()
        self.camera_index = camera_index
        self.capture_count = 0
        self.running = False
        
        # Set up session folder
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name
        self.output_dir = os.path.join(os.path.dirname(__file__), "sessions", session_name)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def list_cameras(self) -> list:
        """List available cameras"""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras
    
    def save_capture(self, frame: np.ndarray):
        """Save a captured frame"""
        self.capture_count += 1
        filename = f"{self.session_name}_{self.capture_count:04d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        cv2.imwrite(filepath, frame, encode_params)
        print(f"Captured: {filename}")
    
    def draw_status(self, frame: np.ndarray, state: State) -> np.ndarray:
        """Draw status overlay on frame"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Status colors
        colors = {
            State.IDLE: (128, 128, 128),
            State.TURNING: (0, 255, 255),
            State.STABILIZING: (255, 128, 0),
            State.CAPTURING: (0, 255, 0),
            State.COOLDOWN: (128, 128, 128),
        }
        color = colors.get(state, (255, 255, 255))
        
        # Draw status bar at bottom
        cv2.rectangle(display, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(display, state.value, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Pages: {self.capture_count}", (w - 150, h - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Delta: {self.detector.last_delta:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Flash green border on capture
        if state == State.CAPTURING:
            cv2.rectangle(display, (0, 0), (w - 1, h - 61), (0, 255, 0), 8)
        
        return display
    
    def run(self):
        """Main loop"""
        print(f"\nPageSnap - Book Scanner")
        print(f"=======================")
        print(f"Session: {self.session_name}")
        print(f"Output:  {self.output_dir}")
        print(f"\nControls:")
        print(f"  - Draw rectangle with mouse to set ROI")
        print(f"  - Press 'r' to reset ROI")
        print(f"  - Press 's' to start/stop detection")
        print(f"  - Press 'q' to quit")
        print(f"\nOpening camera {self.camera_index}...")
        
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            print(f"Available cameras: {self.list_cameras()}")
            return
        
        # Set camera to high resolution if available
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        window_name = "PageSnap"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.roi_selector.mouse_callback)
        cv2.waitKey(1)  # Force window to appear on macOS
        
        detection_active = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Store original for capture
                original_frame = frame.copy()
                
                # Process detection if active
                state = State.IDLE
                should_capture = False
                
                if detection_active:
                    state, should_capture = self.detector.process_frame(
                        frame, self.roi_selector.get_roi()
                    )
                    
                    if should_capture:
                        self.save_capture(original_frame)
                        # Play system beep
                        print("\a", end="", flush=True)
                
                # Draw overlays
                display = self.roi_selector.draw_overlay(frame)
                display = self.draw_status(display, state if detection_active else State.IDLE)
                
                # Show detection status
                status_text = "DETECTING" if detection_active else "PAUSED (press 's' to start)"
                cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if detection_active else (0, 0, 255), 2)
                
                cv2.imshow(window_name, display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.roi_selector.roi = None
                    print("ROI reset")
                elif key == ord('s'):
                    detection_active = not detection_active
                    if detection_active:
                        self.detector.previous_frame = None
                        self.detector.state = State.IDLE
                    print(f"Detection {'started' if detection_active else 'paused'}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nSession ended. Captured {self.capture_count} pages.")
            print(f"Images saved to: {self.output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PageSnap - Automatic page turn detection")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("-n", "--name", type=str, help="Session name (default: timestamp)")
    parser.add_argument("-l", "--list", action="store_true", help="List available cameras")
    
    args = parser.parse_args()
    
    if args.list:
        app = PageSnap()
        cameras = app.list_cameras()
        print(f"Available cameras: {cameras}")
        return
    
    app = PageSnap(camera_index=args.camera, session_name=args.name)
    app.run()


if __name__ == "__main__":
    main()
