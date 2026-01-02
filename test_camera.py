#!/usr/bin/env python3
"""Simple camera test"""
import cv2
import sys

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"Opening camera {camera_index}...")

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Failed to open camera")
    sys.exit(1)

print("Camera opened! Press 'q' to quit.")
print("If no window appears, try running from Terminal.app directly.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done")
