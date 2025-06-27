import cv2
import numpy as np
import platform

# === Config ===
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CROP_TOP = 240  # Crop top 240 pixels → new view is 240
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD = 50
center_x = FRAME_WIDTH // 2


# === Backend ===
backend = cv2.CAP_V4L2 if platform.system() != "Windows" else cv2.CAP_DSHOW

# === Camera Init ===
cap = cv2.VideoCapture(CAMERA_INDEX, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Press 'q' to quit.")

def get_black_mask_and_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return mask, contours

def get_contour_x_bounds(contour):
    xs = contour[:, 0, 0]
    return np.min(xs), np.max(xs)

def get_average_y_near_x(contour, target_x, window=10):
    points = contour[:, 0, :]
    close_points = points[np.abs(points[:, 0] - target_x) < window]
    if len(close_points) == 0:
        return np.mean(points[:, 1])
    return np.mean(close_points[:, 1])

# === Main Loop ===
while True:
    ret, full_frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    # === Crop out top section ===
    frame = full_frame[CROP_TOP:, :]  # frame height is now 380

    mask, contours = get_black_mask_and_contours(frame)
    yellow_bg = np.full_like(frame, YELLOW)
    result = np.where(mask[:, :, None] == 255, frame, yellow_bg)

    # Horizon line based on original frame center (240 - 100 = 140 in cropped)
    horizon_y = FRAME_HEIGHT // 2 - CROP_TOP
    if 0 <= horizon_y < frame.shape[0]:
        cv2.line(result, (0, horizon_y), (FRAME_WIDTH - 1, horizon_y), RED, 2)

    # Draw vertical center line
    center_x = FRAME_WIDTH // 2
    cv2.line(result, (center_x, 0), (center_x, frame.shape[0] - 1), RED, 1)

    # Measure areas below horizon
    error = 0
    left_area = 0
    right_area = 0
    if 0 <= horizon_y < frame.shape[0]:
        lower_mask = np.zeros_like(mask)
        lower_mask[horizon_y:, :] = mask[horizon_y:, :]

        left_area = np.sum(lower_mask[:, :center_x] == 255)
        right_area = np.sum(lower_mask[:, center_x:] == 255)

        error = right_area - left_area

        # Visual indicators
        cv2.rectangle(result, (0, horizon_y), (center_x, frame.shape[0]), (255, 0, 0), 1)
        cv2.rectangle(result, (center_x, horizon_y), (FRAME_WIDTH, frame.shape[0]), (0, 0, 255), 1)

    # === Text overlays ===
    # Error at top-left
    cv2.putText(result, f"Error: {error}", (10, 30), FONT, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, f"Error: {error}", (10, 30), FONT, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Left area at bottom-left
    cv2.putText(result, f"Left: {left_area}", (10, frame.shape[0] - 10), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, f"Left: {left_area}", (10, frame.shape[0] - 10), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Right area at bottom-right
    right_text = f"Right: {right_area}"
    text_size, _ = cv2.getTextSize(right_text, FONT, 0.6, 1)
    right_x = FRAME_WIDTH - text_size[0] - 10
    cv2.putText(result, right_text, (right_x, frame.shape[0] - 10), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, right_text, (right_x, frame.shape[0] - 10), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # === Draw direction arrow ===
    arrow_start = (center_x, frame.shape[0] // 2)
    arrow_color = (255, 255, 255)
    thickness = 4
    tip_length = 0.4
    arrow_len = 50

    if error > 1000:  # Turn left
        arrow_end = (center_x - arrow_len, frame.shape[0] // 2)
    elif error < -1000:  # Turn right
        arrow_end = (center_x + arrow_len, frame.shape[0] // 2)
    else:  # Small error: go forward/down
        arrow_end = (center_x, frame.shape[0] // 2 - arrow_len)

    cv2.arrowedLine(result, arrow_start, arrow_end, arrow_color, thickness, tipLength=tip_length)


    # Show result
    cv2.imshow("Wall Detection + Horizon + Error", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
