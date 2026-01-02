#!/usr/bin/env python3
"""
PageSnap Web - Browser-based page turn detection
"""

import cv2
import numpy as np
import time
import os
import json
import threading
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from flask import Flask, render_template_string, Response, jsonify, request, send_file, url_for

app = Flask(__name__)


class State(Enum):
    IDLE = "Monitoring..."
    TURNING = "Page turning..."
    STABILIZING = "Stabilizing..."
    CAPTURING = "Captured!"
    COOLDOWN = "Cooldown..."


@dataclass
class Config:
    motion_threshold: float = 3.0  # Delta above this = motion detected
    stability_threshold: float = 1.0  # Delta below this = stable
    stability_delay: float = 1.0
    cooldown_period: float = 2.0
    blur_kernel: int = 21
    jpeg_quality: int = 90
    detection_scale: float = 0.25


class PageSnapDetector:
    def __init__(self, config: Config):
        self.config = config
        self.state = State.IDLE
        self.previous_frame: Optional[np.ndarray] = None
        self.stability_start: Optional[float] = None
        self.cooldown_start: Optional[float] = None
        self.last_delta: float = 0.0
    
    def reset(self):
        self.previous_frame = None
        self.state = State.IDLE
        self.stability_start = None
        self.cooldown_start = None
    
    def process_frame(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[State, bool]:
        should_capture = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if roi and roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        small = cv2.resize(gray, None, fx=self.config.detection_scale, fy=self.config.detection_scale)
        blurred = cv2.GaussianBlur(small, (self.config.blur_kernel, self.config.blur_kernel), 0)
        
        if self.previous_frame is None:
            self.previous_frame = blurred
            return self.state, False
        
        diff = cv2.absdiff(blurred, self.previous_frame)
        self.last_delta = np.mean(diff)
        
        now = time.time()
        
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


class PageSnapApp:
    def __init__(self, camera_index: int = 0):
        self.config = Config()
        self.detector = PageSnapDetector(self.config)
        self.camera_index = camera_index
        self.cap = None
        self.detection_active = False
        self.capture_count = 0
        self.roi = None  # (x, y, w, h) as percentages
        self.frame_width = 640
        self.frame_height = 480
        self.lock = threading.Lock()
        self.last_capture_time = 0
        self.ocr_status = self._initial_ocr_status()
        self.ocr_thread: Optional[threading.Thread] = None
        
        # Session setup
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(os.path.dirname(__file__), "sessions", self.session_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _initial_ocr_status(self):
        return {
            'state': 'idle',  # idle, running, complete, error
            'session': None,
            'completed': 0,
            'total': 0,
            'current_page': None,
            'output': None,
            'error': None
        }

    def reset_ocr_status(self):
        self.ocr_status = self._initial_ocr_status()
        self.ocr_thread = None

    def _start_ocr_thread(self, session_name: str, session_path: str, images: list):
        """Start OCR processing in a background thread."""
        def progress_callback(completed, total, current_file):
            with self.lock:
                self.ocr_status.update({
                    'completed': completed,
                    'total': total,
                    'current_page': current_file
                })

        def worker():
            try:
                from ocr_gemini import process_session
                output_path = process_session(
                    session_path,
                    progress_callback=progress_callback,
                    exit_on_error=False
                )
                with self.lock:
                    self.ocr_status.update({
                        'state': 'complete',
                        'output': output_path,
                        'error': None,
                        'completed': len(images),
                        'total': len(images),
                        'current_page': None
                    })
            except Exception as e:
                with self.lock:
                    self.ocr_status.update({
                        'state': 'error',
                        'error': str(e),
                        'current_page': None
                    })

        with self.lock:
            self.ocr_status.update({
                'state': 'running',
                'session': session_name,
                'completed': 0,
                'total': len(images),
                'current_page': None,
                'output': None,
                'error': None
            })

        self.ocr_thread = threading.Thread(target=worker, daemon=True)
        self.ocr_thread.start()

    def trigger_ocr(self, session_name: str):
        """Validate and start OCR for a session if not already running."""
        session_path = os.path.join(os.path.dirname(__file__), "sessions", session_name)
        if not os.path.exists(session_path):
            return False, f"Session not found: {session_name}"

        images = sorted([f for f in os.listdir(session_path) if f.endswith('.jpg')])
        if not images:
            return False, "No images in session"

        with self.lock:
            if self.ocr_status.get('state') == 'running':
                return False, "OCR already running"

        self._start_ocr_thread(session_name, session_path, images)
        return True, None
    
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_roi_pixels(self):
        if not self.roi:
            return None
        x_pct, y_pct, w_pct, h_pct = self.roi
        return (
            int(x_pct * self.frame_width),
            int(y_pct * self.frame_height),
            int(w_pct * self.frame_width),
            int(h_pct * self.frame_height)
        )
    
    def save_capture(self, frame: np.ndarray):
        self.capture_count += 1
        filename = f"{self.session_name}_{self.capture_count:04d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        cv2.imwrite(filepath, frame, encode_params)
        self.last_capture_time = time.time()
        print(f"Captured: {filename}")
    
    def generate_frames(self):
        self.start_camera()
        
        while True:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            original_frame = frame.copy()
            display_frame = frame.copy()
            
            with self.lock:
                state = State.IDLE
                should_capture = False
                
                if self.detection_active:
                    roi_pixels = self.get_roi_pixels()
                    state, should_capture = self.detector.process_frame(frame, roi_pixels)
                    
                    if should_capture:
                        self.save_capture(original_frame)
                
                # Status bar
                h, w = display_frame.shape[:2]
                colors = {
                    State.IDLE: (128, 128, 128),
                    State.TURNING: (0, 255, 255),
                    State.STABILIZING: (255, 128, 0),
                    State.CAPTURING: (0, 255, 0),
                    State.COOLDOWN: (128, 128, 128),
                }
                color = colors.get(state, (255, 255, 255))
                
                cv2.rectangle(display_frame, (0, h - 50), (w, h), (0, 0, 0), -1)
                status = state.value if self.detection_active else "PAUSED"
                cv2.putText(display_frame, status, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display_frame, f"Pages: {self.capture_count}", (w - 150, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Flash on capture
                if time.time() - self.last_capture_time < 0.3:
                    cv2.rectangle(display_frame, (0, 0), (w - 1, h - 51), (0, 255, 0), 8)
            
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


page_snap = PageSnapApp(camera_index=0)


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Page Snap — Doodle Reader</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --ink: #1a1a1a;
            --ink-soft: #3d3d3d;
            --ink-muted: #6b6b6b;
            --cream: #faf8f5;
            --cream-warm: #f5f2ed;
            --cream-dark: #ebe7e0;
            --surface: #ffffff;
            --border: #d4d0c8;
            --accent: #4f46e5;
            --accent-muted: #6366f1;
            --accent-soft: #e0e7ff;
            --status-success: #16a34a;
            --status-error: #dc2626;
            --status-warning: #f59e0b;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--cream);
            color: var(--ink);
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
        .back-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            color: var(--accent);
            text-decoration: none;
            margin-bottom: 16px;
            font-weight: 500;
        }
        .back-link:hover { text-decoration: underline; }
        .header {
            display: flex;
            align-items: baseline;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 2px solid var(--border);
        }
        .header h1 {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 28px;
            font-weight: 600;
            color: var(--ink);
        }
        .header .subtitle {
            font-size: 14px;
            color: var(--ink-muted);
            margin-left: 12px;
            font-weight: 500;
        }
        .video-container {
            position: relative;
            background: var(--ink);
            margin-bottom: 20px;
            border: 2px solid var(--ink);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 4px 4px 0 var(--ink);
        }
        #video-feed {
            width: 100%;
            display: block;
        }
        .controls {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            border: 2px solid var(--ink);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
            font-family: 'Inter', sans-serif;
        }
        .btn-start {
            background: var(--accent);
            color: white;
            min-width: 160px;
            box-shadow: 3px 3px 0 var(--ink);
        }
        .btn-start:hover {
            transform: translate(-2px, -2px);
            box-shadow: 5px 5px 0 var(--ink);
        }
        .btn-start:active {
            transform: translate(2px, 2px);
            box-shadow: 1px 1px 0 var(--ink);
        }
        .btn-start.active { background: var(--status-error); }
        .btn-secondary {
            background: var(--surface);
            color: var(--ink);
            box-shadow: 2px 2px 0 var(--ink);
        }
        .btn-secondary:hover {
            background: var(--cream);
            transform: translate(-1px, -1px);
            box-shadow: 3px 3px 0 var(--ink);
        }
        .btn-ocr {
            background: var(--status-success) !important;
            color: white;
        }
        select {
            padding: 12px 16px;
            font-size: 14px;
            border-radius: 6px;
            background: var(--surface);
            color: var(--ink);
            border: 2px solid var(--border);
            font-family: 'Inter', sans-serif;
            cursor: pointer;
        }
        select:hover { border-color: var(--ink); }
        .status {
            padding: 20px;
            background: var(--surface);
            border: 2px solid var(--border);
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            font-size: 14px;
        }
        .status-row:last-child { margin-bottom: 0; }
        .status-row span:first-child { color: var(--ink-muted); }
        .status-row span:last-child { font-weight: 600; font-family: 'SF Mono', ui-monospace, monospace; font-size: 13px; }
        .settings {
            background: var(--cream-warm);
            padding: 24px;
            border: 2px solid var(--border);
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .settings h3 {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--ink);
        }
        .setting-row {
            display: flex;
            align-items: center;
            margin-bottom: 16px;
        }
        .setting-row:last-child { margin-bottom: 0; }
        .setting-row label {
            width: 160px;
            flex-shrink: 0;
            font-size: 14px;
            font-weight: 500;
        }
        .setting-row input[type="range"] {
            flex: 1;
            margin: 0 16px;
            accent-color: var(--accent);
        }
        .setting-row .value {
            width: 50px;
            text-align: right;
            font-weight: 600;
            font-family: 'SF Mono', ui-monospace, monospace;
            font-size: 14px;
        }
        .setting-help {
            font-size: 12px;
            color: var(--ink-muted);
            margin-top: 6px;
        }
        .instructions {
            background: var(--surface);
            padding: 20px;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            color: var(--ink-soft);
        }
        .instructions h3 {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--ink);
        }
        .instructions ol { margin-left: 20px; }
        .instructions li { margin-bottom: 8px; line-height: 1.5; }
        #capture-flash {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(79, 70, 229, 0.3);
            pointer-events: none;
            z-index: 1000;
        }
        .capture-sound { display: none; }
    </style>
</head>
<body>
    <div id="capture-flash"></div>
    <audio id="capture-sound" class="capture-sound" preload="auto">
        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdH2LkpONgHBkZHN/ipSUjoF0aGZue4qUlo+DdsHBwb2/vry+vMDAvL+/vb++vby8u76+vL6+vb6+vb2+vr6+vb69vb29vry8vL28vb28vLy7u7y8vLy7" type="audio/wav">
    </audio>
    <div class="container">
        <a href="http://localhost:3001" class="back-link">← Back to Doodle Reader</a>

        <div class="header">
            <h1>Page Snap</h1>
            <span class="subtitle">Camera OCR Utility</span>
        </div>

        <div class="video-container" id="video-container">
            <img id="video-feed" src="/video_feed" alt="Camera Feed">
        </div>

        <div class="controls">
            <button id="toggle-btn" class="btn-start" onclick="toggleDetection()">Start Scanning</button>
            <button class="btn-secondary" onclick="newSession()">New Session</button>
            <button class="btn-secondary btn-ocr" onclick="runOCR()" id="ocr-btn">Run OCR</button>
            <select id="camera-select" onchange="switchCamera(this.value)">
                <option value="">Loading cameras...</option>
            </select>
            <select id="session-select" onchange="loadSession(this.value)">
                <option value="">Select Session...</option>
            </select>
        </div>

        <div class="status">
            <div class="status-row">
                <span>Session:</span>
                <span id="session-name">{{ session_name }}</span>
            </div>
            <div class="status-row">
                <span>Pages Captured:</span>
                <span id="capture-count">0</span>
            </div>
            <div class="status-row">
                <span>Output:</span>
                <span id="output-dir">{{ output_dir }}</span>
            </div>
            <div class="status-row" style="margin-top: 15px; padding-top: 12px; border-top: 1px solid var(--border);">
                <span>State:</span>
                <span id="current-state">IDLE</span>
            </div>
            <div class="status-row">
                <span>Motion Level:</span>
                <span id="delta-display">0</span>
            </div>
            <div class="status-row" id="ocr-status-row" style="margin-top: 15px; padding-top: 12px; border-top: 1px solid var(--border);">
                <span>OCR:</span>
                <span id="ocr-status">Idle</span>
            </div>
            <div class="status-row" id="ocr-progress-row" style="display: none;">
                <span>OCR Progress:</span>
                <span id="ocr-progress">0/0</span>
            </div>
            <div class="status-row" id="ocr-output-row" style="display: none;">
                <span>OCR Output:</span>
                <span><a id="ocr-output-link" href="#" target="_blank">Open OCR markdown</a></span>
            </div>
        </div>

        <div class="settings">
            <h3>Settings</h3>

            <div class="setting-row">
                <label>Sensitivity:</label>
                <input type="range" id="sensitivity" min="1" max="10" value="5" onchange="updateSensitivity(this.value)">
                <span class="value" id="sensitivity-val">5</span>
            </div>
            <div class="setting-help" style="margin-left: 160px; margin-bottom: 16px;">
                Higher = detects smaller movements. Lower = needs bigger page flips.
            </div>

            <div class="setting-row">
                <label>Capture Delay:</label>
                <input type="range" id="stability-delay" min="0.3" max="3" step="0.1" value="1.0" onchange="updateDelay(this.value)">
                <span class="value" id="stability-delay-val">1.0s</span>
            </div>
            <div class="setting-help" style="margin-left: 160px;">
                How long the page must be still before capturing. Increase if capturing too early.
            </div>
        </div>

        <div class="instructions">
            <h3>How to Use</h3>
            <ol>
                <li>Position your camera/phone overhead pointing at the book</li>
                <li>Click "Start Scanning"</li>
                <li>Turn pages - the app captures automatically when the page settles</li>
                <li>Images save to the sessions folder for OCR processing</li>
            </ol>
        </div>
    </div>
    
    <script>
        let isDetecting = false;
        let lastOcrState = null;
        
        // Load available cameras on page load
        function loadCameras() {
            fetch('/list_cameras')
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('camera-select');
                    select.innerHTML = data.cameras.map(i => 
                        `<option value="${i}" ${i === data.current ? 'selected' : ''}>Camera ${i}</option>`
                    ).join('');
                });
        }
        
        function loadSessions() {
            fetch('/list_sessions')
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('session-select');
                    select.innerHTML = '<option value="">Select Session...</option>' + 
                        data.sessions.map(s => 
                            `<option value="${s.name}" ${s.name === data.current ? 'selected' : ''}>${s.name} (${s.image_count} pages${s.has_ocr ? ', OCR done' : ''})</option>`
                        ).join('');
                });
        }
        
        function loadSession(name) {
            if (!name) return;
            document.getElementById('session-name').textContent = name;
        }
        
        function runOCR() {
            const sessionSelect = document.getElementById('session-select');
            const sessionName = sessionSelect.value || document.getElementById('session-name').textContent;
            
            if (!sessionName) {
                alert('No session selected');
                return;
            }
            
            const btn = document.getElementById('ocr-btn');
            btn.textContent = 'Starting OCR...';
            btn.disabled = true;
            
            fetch('/run_ocr', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session: sessionName})
            })
            .then(r => r.json())
            .then(data => {
                btn.textContent = 'Run OCR';
                btn.disabled = false;
                if (data.error) {
                    alert('OCR Error: ' + data.error);
                } else {
                    lastOcrState = null; // Let status poll handle updates
                }
            })
            .catch(err => {
                btn.textContent = 'Run OCR';
                btn.disabled = false;
                alert('OCR failed: ' + err);
            });
        }

        loadCameras();
        loadSessions();
        
        function switchCamera(index) {
            if (index === '') return;
            fetch('/set_camera', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({camera: index})
            }).then(() => {
                // Reset button state
                isDetecting = false;
                const btn = document.getElementById('toggle-btn');
                btn.textContent = 'Start Scanning';
                btn.classList.remove('active');
                // Reload video feed
                document.getElementById('video-feed').src = '/video_feed?' + Date.now();
            });
        }
        
        function toggleDetection() {
            fetch('/toggle_detection', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    isDetecting = data.active;
                    const btn = document.getElementById('toggle-btn');
                    btn.textContent = isDetecting ? 'Stop Scanning' : 'Start Scanning';
                    btn.classList.toggle('active', isDetecting);
                    if (isDetecting) {
                        // Reset OCR UI when starting new scan
                        lastOcrState = null;
                        document.getElementById('ocr-status').textContent = 'Idle';
                        document.getElementById('ocr-progress-row').style.display = 'none';
                        document.getElementById('ocr-output-row').style.display = 'none';
                    }
                });
        }
        
        function newSession() {
            fetch('/new_session', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    document.getElementById('session-name').textContent = data.session_name;
                    document.getElementById('output-dir').textContent = data.output_dir;
                    document.getElementById('capture-count').textContent = '0';
                    // Reset button state since detection was stopped
                    isDetecting = false;
                    const btn = document.getElementById('toggle-btn');
                    btn.textContent = 'Start Scanning';
                    btn.classList.remove('active');
                    lastOcrState = null;
                    document.getElementById('ocr-status').textContent = 'Idle';
                    document.getElementById('ocr-progress-row').style.display = 'none';
                    document.getElementById('ocr-output-row').style.display = 'none';
                });
        }
        
        function updateSensitivity(value) {
            document.getElementById('sensitivity-val').textContent = value;
            // Convert 1-10 scale to thresholds (inverted: higher sensitivity = lower threshold)
            // Based on observed values: motion ~10, still ~0.4
            const motionThreshold = 11 - value;  // 10 down to 1
            const stabilityThreshold = 2.2 - (value * 0.2);  // 2.0 down to 0.2
            
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: 'motion_threshold', value: motionThreshold})
            });
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: 'stability_threshold', value: stabilityThreshold})
            });
        }
        
        function updateDelay(value) {
            document.getElementById('stability-delay-val').textContent = value + 's';
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: 'stability_delay', value: parseFloat(value)})
            });
        }
        
        function updateOcrUI(ocr) {
            const statusEl = document.getElementById('ocr-status');
            const progressRow = document.getElementById('ocr-progress-row');
            const progressText = document.getElementById('ocr-progress');
            const outputRow = document.getElementById('ocr-output-row');
            const outputLink = document.getElementById('ocr-output-link');
            const ocrBtn = document.getElementById('ocr-btn');

            if (!ocr) return;

            let statusText = 'Idle';
            if (ocr.state === 'running') statusText = 'Running...';
            else if (ocr.state === 'complete') statusText = 'Complete';
            else if (ocr.state === 'error') statusText = 'Error: ' + (ocr.error || 'Unknown');
            statusEl.textContent = statusText;

            if (ocr.state === 'running') {
                progressRow.style.display = 'flex';
                const total = ocr.total || 0;
                const completed = ocr.completed || 0;
                const current = ocr.current_page ? ` (${ocr.current_page})` : '';
                progressText.textContent = total ? `${completed}/${total} pages${current}` : 'Preparing...';
                outputRow.style.display = 'none';
            } else if (ocr.state === 'complete' && ocr.output_url) {
                progressRow.style.display = 'none';
                outputRow.style.display = 'flex';
                outputLink.href = ocr.output_url;
                outputLink.textContent = 'Open OCR markdown';
            } else {
                progressRow.style.display = 'none';
                outputRow.style.display = 'none';
            }

            if (ocrBtn) {
                ocrBtn.disabled = ocr.state === 'running';
                ocrBtn.textContent = ocr.state === 'running' ? 'OCR in progress...' : 'Run OCR';
            }
        }
        
        // Poll for status updates
        let lastCount = 0;
        setInterval(() => {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('capture-count').textContent = data.capture_count;
                    document.getElementById('current-state').textContent = data.state;
                    document.getElementById('delta-display').textContent = 
                        data.delta + ' (trigger: >' + data.motion_threshold + ', stable: <' + data.stability_threshold + ')';
                    
                    // Flash and sound on new capture
                    if (data.capture_count > lastCount) {
                        const flash = document.getElementById('capture-flash');
                        flash.style.display = 'block';
                        setTimeout(() => flash.style.display = 'none', 200);
                        
                        // Try to play sound
                        try {
                            document.getElementById('capture-sound').play();
                        } catch(e) {}
                    }
                    lastCount = data.capture_count;
                    
                    if (data.ocr) {
                        updateOcrUI(data.ocr);
                        if (lastOcrState !== 'complete' && data.ocr.state === 'complete') {
                            loadSessions();
                        }
                        lastOcrState = data.ocr.state;
                    }
                });
        }, 500);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, 
                                  session_name=page_snap.session_name,
                                  output_dir=page_snap.output_dir,
                                  current_camera=page_snap.camera_index)


@app.route('/list_cameras')
def list_cameras():
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
            cap.release()
    return jsonify({'cameras': cameras, 'current': page_snap.camera_index})


@app.route('/set_camera', methods=['POST'])
def set_camera():
    data = request.json
    new_index = int(data['camera'])
    with page_snap.lock:
        page_snap.detection_active = False
        if page_snap.cap:
            page_snap.cap.release()
            page_snap.cap = None
        page_snap.camera_index = new_index
        page_snap.detector.reset()
    return jsonify({'ok': True, 'camera': new_index})


@app.route('/video_feed')
def video_feed():
    return Response(page_snap.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    with page_snap.lock:
        page_snap.detection_active = not page_snap.detection_active
        if page_snap.detection_active:
            page_snap.detector.reset()
            page_snap.reset_ocr_status()  # Clear stale OCR status when starting
    return jsonify({'active': page_snap.detection_active})


@app.route('/set_roi', methods=['POST'])
def set_roi():
    data = request.json
    with page_snap.lock:
        page_snap.roi = (data['x'], data['y'], data['w'], data['h'])
    return jsonify({'ok': True})


@app.route('/clear_roi', methods=['POST'])
def clear_roi():
    with page_snap.lock:
        page_snap.roi = None
    return jsonify({'ok': True})


@app.route('/new_session', methods=['POST'])
def new_session():
    with page_snap.lock:
        page_snap.detection_active = False  # Stop detection
        page_snap.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        page_snap.output_dir = os.path.join(os.path.dirname(__file__), "sessions", page_snap.session_name)
        os.makedirs(page_snap.output_dir, exist_ok=True)
        page_snap.capture_count = 0
        page_snap.detector.reset()
        page_snap.reset_ocr_status()
    return jsonify({'session_name': page_snap.session_name, 'output_dir': page_snap.output_dir, 'detection_stopped': True})


@app.route('/update_setting', methods=['POST'])
def update_setting():
    data = request.json
    with page_snap.lock:
        setattr(page_snap.config, data['name'], data['value'])
        page_snap.detector.config = page_snap.config
    return jsonify({'ok': True})


@app.route('/status')
def status():
    with page_snap.lock:
        ocr_status = dict(page_snap.ocr_status)
        capture_count = page_snap.capture_count
        detection_active = page_snap.detection_active
        state_value = page_snap.detector.state.value
        last_delta = round(page_snap.detector.last_delta, 1)
        motion_threshold = page_snap.config.motion_threshold
        stability_threshold = page_snap.config.stability_threshold

    if ocr_status.get('output') and ocr_status.get('session') and os.path.exists(ocr_status['output']):
        ocr_status['output_url'] = url_for('session_ocr', session_name=ocr_status['session'])

    return jsonify({
        'capture_count': capture_count,
        'detection_active': detection_active,
        'state': state_value,
        'delta': last_delta,
        'motion_threshold': motion_threshold,
        'stability_threshold': stability_threshold,
        'ocr': ocr_status
    })


@app.route('/list_sessions')
def list_sessions():
    """List all available sessions."""
    sessions_dir = os.path.join(os.path.dirname(__file__), "sessions")
    sessions = []
    if os.path.exists(sessions_dir):
        for name in sorted(os.listdir(sessions_dir), reverse=True):
            session_path = os.path.join(sessions_dir, name)
            if os.path.isdir(session_path):
                images = [f for f in os.listdir(session_path) if f.endswith('.jpg')]
                sessions.append({
                    'name': name,
                    'image_count': len(images),
                    'has_ocr': os.path.exists(os.path.join(session_path, f"{name}_ocr.md"))
                })
    return jsonify({'sessions': sessions, 'current': page_snap.session_name})


@app.route('/run_ocr', methods=['POST'])
def run_ocr():
    """Run OCR on a session using Gemini 2.0 Flash."""
    data = request.json or {}
    session_name = data.get('session') or page_snap.session_name

    started, error = page_snap.trigger_ocr(session_name)
    if error:
        code = 404 if "Session not found" in error else 409 if "already running" in error else 400
        return jsonify({'error': error}), code

    return jsonify({'ok': True})


@app.route('/session_ocr/<session_name>')
def session_ocr(session_name):
    """Serve the OCR markdown for a session."""
    session_path = os.path.join(os.path.dirname(__file__), "sessions", session_name)
    md_path = os.path.join(session_path, f"{session_name}_ocr.md")

    if not os.path.exists(md_path):
        return jsonify({'error': 'OCR output not found'}), 404

    return send_file(md_path, mimetype='text/markdown', download_name=f"{session_name}_ocr.md")


@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """Export session images as a single PDF."""
    data = request.json
    session_name = data.get('session', page_snap.session_name)
    session_path = os.path.join(os.path.dirname(__file__), "sessions", session_name)

    if not os.path.exists(session_path):
        return jsonify({'error': f'Session not found: {session_name}'}), 404

    # Get all images sorted by name
    images = sorted([f for f in os.listdir(session_path) if f.endswith('.jpg')])
    if not images:
        return jsonify({'error': 'No images in session'}), 400

    try:
        from PIL import Image

        # Load all images and convert to RGB
        image_list = []
        for img_name in images:
            img_path = os.path.join(session_path, img_name)
            img = Image.open(img_path).convert('RGB')
            image_list.append(img)

        # Create PDF
        pdf_path = os.path.join(session_path, f"{session_name}.pdf")
        if len(image_list) > 0:
            image_list[0].save(
                pdf_path,
                save_all=True,
                append_images=image_list[1:] if len(image_list) > 1 else [],
                resolution=100.0
            )

        return jsonify({
            'ok': True,
            'output': pdf_path,
            'page_count': len(images)
        })
    except ImportError:
        return jsonify({'error': 'Pillow not installed. Run: pip install Pillow'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_pdf/<session_name>')
def download_pdf(session_name):
    """Download the PDF for a session."""
    session_path = os.path.join(os.path.dirname(__file__), "sessions", session_name)
    pdf_path = os.path.join(session_path, f"{session_name}.pdf")

    if not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF not found. Export first.'}), 404

    return send_file(pdf_path, as_attachment=True, download_name=f"{session_name}.pdf")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', type=int, default=0)
    parser.add_argument('-p', '--port', type=int, default=5000)
    args = parser.parse_args()
    
    page_snap.camera_index = args.camera
    print(f"\nPageSnap Web")
    print(f"============")
    print(f"Camera: {args.camera}")
    print(f"Open http://localhost:{args.port} in your browser\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
