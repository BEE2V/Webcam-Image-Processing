# Working wall mask
```python
import cv2
import numpy as np
import platform

# Config
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YELLOW = (0, 255, 255)  # BGR format for bright yellow

# Select backend
if platform.system() == "Windows":
    backend = cv2.CAP_DSHOW
else:
    backend = cv2.CAP_V4L2

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX, backend)
if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create masks for black and white regions
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combine masks
    combined_mask = cv2.bitwise_or(black_mask, white_mask)

    # Create a yellow background image
    yellow_bg = np.full_like(frame, YELLOW, dtype=np.uint8)

    # Apply masks: keep wall and floor from original, rest from yellow
    result = np.where(combined_mask[:, :, np.newaxis] == 255, frame, yellow_bg)

    # Show the result
    cv2.imshow("Black Wall & White Floor on Yellow", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

```

# Edge detection working
```python
import cv2
import platform

# Configuration
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Set appropriate backend
if platform.system() == "Windows":
    backend = cv2.CAP_DSHOW
else:
    backend = cv2.CAP_V4L2

# Open USB camera
cap = cv2.VideoCapture(CAMERA_INDEX, backend)
if not cap.isOpened():
    print("Error: Could not open the USB camera.")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to grab frame.")
        break

    # Edge detection
    edges = cv2.Canny(frame, 100, 200)

    # Display edge-detected frame
    cv2.imshow("Edge Detection", edges)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
```