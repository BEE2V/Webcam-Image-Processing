import cv2

print("Searching for available cameras...")

for index in range(500):
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        print(f"Camera found at index {index}")
        cap.release()
    else:
        print(f"No camera at index {index}")
