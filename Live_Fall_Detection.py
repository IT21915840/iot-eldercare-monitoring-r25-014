from keras.models import load_model
import cv2
import numpy as np
from datetime import datetime

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

np.set_printoptions(suppress=True)

model = load_model("Fall_detection/keras_Model.h5", compile=False)

class_names = open("Fall_detection/labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

print(f"{'Timestamp':<28} | {'Status / Class':<30} | {'Confidence':<15}")
print(f"{'-' * 28:<28} | {'-' * 30:<30} | {'-' * 15:<15}")

while True:
    ret, image = camera.read()

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    (boxes, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    if len(boxes) > 0:
        image_for_model = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        image_for_model = np.asarray(image_for_model, dtype=np.float32).reshape(1, 224, 224, 3)
        image_for_model = (image_for_model / 127.5) - 1

        prediction = model.predict(image_for_model, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        class_name_clean = class_name[2:].strip()
        confidence_str = f"{confidence_score * 100:.2f}%"
        print(f"{timestamp_str:<28} | {class_name_clean:<30} | {confidence_str:<15}")

    else:
        print(f"{timestamp_str:<28} | {'No person detected':<30} | {'-':<15}")

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

print("\nExiting...")
camera.release()