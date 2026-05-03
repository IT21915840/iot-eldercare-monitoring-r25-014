from keras.models import load_model
import cv2
import numpy as np
from datetime import datetime
import time

# Load HOG for comparison
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

np.set_printoptions(suppress=True)

print("Loading Fall Detection Model...")
model = load_model("Fall_detection/keras_Model.h5", compile=False)
class_names = open("Fall_detection/labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

print("\n=== Standalone Fall Detection Tester ===")
print("Opening camera window... Press 'ESC' on the window to exit.")

fall_start_time = None

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 1. AI Fall Detection (No HOG Gating)
    image_for_model = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_for_model = np.asarray(image_for_model, dtype=np.float32).reshape(1, 224, 224, 3)
    image_for_model = (image_for_model / 127.5) - 1

    prediction = model.predict(image_for_model, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index] * 100

    # Normalize label
    raw_upper = class_name.upper()
    if "NOT" in raw_upper or "NO" in raw_upper:
        fall_status = "Normal"
    elif "FALL" in raw_upper:
        fall_status = "FALL"
    else:
        fall_status = class_name

    # 2. Persistence Check (3s)
    is_falling = "fall" in fall_status.lower()
    
    if is_falling:
        if fall_start_time is None:
            fall_start_time = time.time()
        elapsed = time.time() - fall_start_time
        if elapsed >= 3.0:
            display_status = f"CRITICAL FALL DETECTED ({confidence_score:.1f}%)"
            color = (0, 0, 255) # Red
        else:
            display_status = f"POTENTIAL FALL... Hold ({3.0 - elapsed:.1f}s)"
            color = (0, 165, 255) # Orange
    else:
        fall_start_time = None
        display_status = f"Status: Normal ({confidence_score:.1f}%)"
        color = (0, 255, 0) # Green

    # Draw status on the video feed
    cv2.putText(frame, display_status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Fall Detection Test", frame)

    if cv2.waitKey(1) == 27: # ESC key to close
        break

camera.release()
cv2.destroyAllWindows()
print("Tester closed.")