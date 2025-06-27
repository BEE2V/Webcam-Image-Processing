import cv2
import numpy as np
import platform

# === CONFIG ===
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CROP_TOP = 100  # Crop top 100 pixels
GRID_SIZE = 10  # Grid spacing in cm
FLOOR_WIDTH = 40   # Real-world width in cm
FLOOR_HEIGHT = 80  # Real-world depth in cm

# === Setup backend ===
backend = cv2.CAP_V4L2 if platform.system() != "Windows" else cv2.CAP_DSHOW

# === Open camera ===
cap = cv2.VideoCapture(CAMERA_INDEX, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Click 4 floor corners: bottom-left, bottom-right, top-right, top-left.")

# === Global Variables for Mouse Callback ===
clicked_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"✅ Point {len(clicked_points)}: ({x}, {y})")

cv2.namedWindow("Select Floor Corners")
cv2.setMouseCallback("Select Floor Corners", mouse_callback)

# === Wait for 4 corner clicks on the cropped frame ===
while len(clicked_points) < 4:
    ret, full_frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    cropped = full_frame[CROP_TOP:, :].copy()
    for pt in clicked_points:
        cv2.circle(cropped, pt, 5, (0, 0, 255), -1)
    cv2.imshow("Select Floor Corners", cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyWindow("Select Floor Corners")

# === Define real-world floor rectangle (in cm) ===
world_pts = np.float32([
    [0, FLOOR_HEIGHT],             # bottom-left
    [FLOOR_WIDTH, FLOOR_HEIGHT],   # bottom-right
    [FLOOR_WIDTH, 0],              # top-right
    [0, 0]                         # top-left
])

# === Compute homography from real world to cropped image ===
image_pts = np.float32(clicked_points)
H, _ = cv2.findHomography(world_pts, image_pts)

print("✅ Grid overlay running. Press 'q' to quit.")

# === Main Loop ===
while True:
    ret, full_frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    cropped = full_frame[CROP_TOP:, :].copy()

    # === Draw grid ===
    for x in range(0, FLOOR_WIDTH + 1, GRID_SIZE):
        for y in range(0, FLOOR_HEIGHT + 1, GRID_SIZE):
            pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)
            x_img, y_img = img_pt[0][0].astype(int)
            if 0 <= x_img < FRAME_WIDTH and 0 <= y_img < cropped.shape[0]:
                cv2.circle(cropped, (x_img, y_img), 2, (255, 255, 255), -1)

    # === Draw grid lines (horizontal) ===
    for y in range(0, FLOOR_HEIGHT + 1, GRID_SIZE):
        row_pts = []
        for x in range(0, FLOOR_WIDTH + 1, GRID_SIZE):
            pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)[0][0]
            row_pts.append((int(img_pt[0]), int(img_pt[1])))
        for i in range(len(row_pts) - 1):
            cv2.line(cropped, row_pts[i], row_pts[i+1], (100, 255, 100), 1)

    # === Draw grid lines (vertical) ===
    for x in range(0, FLOOR_WIDTH + 1, GRID_SIZE):
        col_pts = []
        for y in range(0, FLOOR_HEIGHT + 1, GRID_SIZE):
            pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)[0][0]
            col_pts.append((int(img_pt[0]), int(img_pt[1])))
        for i in range(len(col_pts) - 1):
            cv2.line(cropped, col_pts[i], col_pts[i+1], (100, 255, 100), 1)

    cv2.imshow("Floor Grid Overlay", cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
