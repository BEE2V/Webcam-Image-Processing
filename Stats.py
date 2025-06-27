import cv2
import numpy as np
import platform
import psutil, os
import time

# === Config ===
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
THRESHOLD = 50
TARGET_FPS = 10
SLEEP_TIME = 1 / TARGET_FPS

# === Set backend ===
backend = cv2.CAP_V4L2 if platform.system() != "Windows" else cv2.CAP_DSHOW

# === Initialize Camera ===
cap = cv2.VideoCapture(CAMERA_INDEX, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

# === Process Info for self-monitoring ===
proc = psutil.Process(os.getpid())

def get_black_mask_and_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return mask, contours

print("✅ Press 'q' to quit.")

# === Main Loop ===
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    # Downsample for processing
    small_frame = cv2.resize(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))

    mask, contours = get_black_mask_and_contours(small_frame)
    yellow_bg = np.full_like(small_frame, YELLOW)
    result = np.where(mask[:, :, None] == 255, small_frame, yellow_bg)

    if contours:
        cv2.drawContours(result, [contours[0]], -1, GREEN, 1)

    # === Self CPU & RAM usage ===
    ram_usage = proc.memory_info().rss / (1024 * 1024)  # in MB
    cpu_usage = proc.cpu_percent(interval=0.0)          # in %

    info = f"CPU: {cpu_usage:.1f}%  RAM: {ram_usage:.0f}MB"
    cv2.putText(result, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    # Upscale for display
    display_frame = cv2.resize(result, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.imshow("Black Wall with Self Usage", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - start_time
    if elapsed < SLEEP_TIME:
        time.sleep(SLEEP_TIME - elapsed)

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
