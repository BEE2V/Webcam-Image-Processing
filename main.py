import cv2
import numpy as np
import platform

# === CONFIG ===
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CROP_TOP = 100  # Crop top 100 pixels

GRID_SIZE = 1  # Grid spacing in cm (smaller for detailed squares)
SQUARE_WIDTH = 7   # Real-world width of square in cm
SQUARE_HEIGHT = 7  # Real-world height of square in cm
GRID_EXTEND = 7     # How many squares to extend in all directions

# === Setup backend ===
backend = cv2.CAP_V4L2 if platform.system() != "Windows" else cv2.CAP_DSHOW

# === Open camera ===
cap = cv2.VideoCapture(CAMERA_INDEX, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Click 4 square corners: bottom-left, bottom-right, top-right, top-left.")

# === Global Variables for Mouse Callback ===
clicked_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"✅ Point {len(clicked_points)}: ({x}, {y})")

cv2.namedWindow("Select Square Corners")
cv2.setMouseCallback("Select Square Corners", mouse_callback)

# === Wait for 4 corner clicks on the cropped frame ===
while len(clicked_points) < 4:
    ret, full_frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    cropped = full_frame[CROP_TOP:, :].copy()
    for pt in clicked_points:
        cv2.circle(cropped, pt, 5, (0, 0, 255), -1)
    cv2.imshow("Select Square Corners", cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyWindow("Select Square Corners")

# === Define real-world square (in cm) ===
world_pts = np.float32([
    [0, SQUARE_HEIGHT],               # bottom-left
    [SQUARE_WIDTH, SQUARE_HEIGHT],   # bottom-right
    [SQUARE_WIDTH, 0],               # top-right
    [0, 0]                            # top-left
])

# === Compute homography from real world to cropped image ===
image_pts = np.float32(clicked_points)
H, _ = cv2.findHomography(world_pts, image_pts)

print("✅ Grid overlay running. Press 'q' to quit.")

# === Grid range with extension ===
x_start = -GRID_EXTEND * GRID_SIZE
x_end = (SQUARE_WIDTH + GRID_EXTEND) + 1
y_start = -GRID_EXTEND * GRID_SIZE
y_end = (SQUARE_HEIGHT + GRID_EXTEND) + 1

# === Main Loop ===
while True:
    ret, full_frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    cropped = full_frame[CROP_TOP:, :].copy()

    # === Draw grid points ===
    for x in range(x_start, x_end, GRID_SIZE):
        for y in range(y_start, y_end, GRID_SIZE):
            pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)
            x_img, y_img = img_pt[0][0].astype(int)
            if 0 <= x_img < FRAME_WIDTH and 0 <= y_img < cropped.shape[0]:
                cv2.circle(cropped, (x_img, y_img), 2, (255, 255, 255), -1)

    # === Draw grid lines (horizontal) ===
    for y in range(y_start, y_end, GRID_SIZE):
        row_pts = []
        for x in range(x_start, x_end, GRID_SIZE):
            pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)[0][0]
            row_pts.append((int(img_pt[0]), int(img_pt[1])))
        for i in range(len(row_pts) - 1):
            cv2.line(cropped, row_pts[i], row_pts[i+1], (100, 255, 100), 1)

    # === Draw grid lines (vertical) ===
    for x in range(x_start, x_end, GRID_SIZE):
        col_pts = []
        for y in range(y_start, y_end, GRID_SIZE):
            pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)[0][0]
            col_pts.append((int(img_pt[0]), int(img_pt[1])))
        for i in range(len(col_pts) - 1):
            cv2.line(cropped, col_pts[i], col_pts[i+1], (100, 255, 100), 1)

    # === Show grid overlay ===
    cv2.imshow("Square Grid Overlay", cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
