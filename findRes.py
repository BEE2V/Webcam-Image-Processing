import cv2
import platform
import time

# === Configuration ===
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detect OS and set backend
if platform.system() == "Windows":
    backend = cv2.CAP_DSHOW
else:
    backend = cv2.CAP_V4L2

# === Try to open the camera ===
print(f"Opening camera at index {CAMERA_INDEX}...")
start_time = time.time()

cap = cv2.VideoCapture(CAMERA_INDEX, backend)
open_time = time.time() - start_time

if not cap.isOpened():
    print(f"❌ Failed to open camera at index {CAMERA_INDEX}")
    exit()

print(f"✅ Camera opened successfully in {open_time:.2f} seconds")

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Print actual resolution
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Using resolution: {int(actual_width)}x{int(actual_height)}")

# === Main processing loop ===
print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        break

    # Basic processing: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display original and processed views
    cv2.imshow("USB Camera - Original", frame)
    cv2.imshow("USB Camera - Grayscale", gray)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")
