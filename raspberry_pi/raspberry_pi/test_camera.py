import cv2

print("Scanning for cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Index {i}: No camera")
    else:
        ret, frame = cap.read()
        if ret:
            print(f"Index {i}: SUCCESS! Found a working camera.")
        else:
            print(f"Index {i}: Opened, but failed to read frame.")
        cap.release()
print("Scan complete.")
