import tensorflow as tf
import numpy as np
import cv2
import os
import time
from datetime import datetime
from tensorflow.keras.models import model_from_json

mapper = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    raise FileNotFoundError(f"Cannot load Haar cascade from {HAAR_CASCADE_PATH}")
print("Haar Cascade loaded successfully.")

try:
    with open("Emotional/model.yaml", "r") as json_file:
        model_json = json_file.read()
    with tf.keras.utils.custom_object_scope({'Functional': tf.keras.Model}):
        model = model_from_json(model_json)
    model.load_weights("Emotional/model.h5")
    print("Emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


def predict_emotion(face_roi, model, label_mapper):
    if len(face_roi.shape) == 3:
        img_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = face_roi

    img_resized = cv2.resize(img_gray, (48, 48))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_rgb = cv2.cvtColor((img_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img_final = np.expand_dims(img_rgb, axis=0)
    predictions = model.predict(img_final, verbose=0)
    predicted_index = np.argmax(predictions[0])
    return label_mapper.get(predicted_index, "Unknown"), np.max(predictions[0]) * 100


def webcam_emotion_cli(model, label_mapper, face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    header = (
        "==========================================================================\n"
        "| TIME                 | FACE (x,y,w,h)       | EMOTION       | CONFIDENCE |\n"
        "=========================================================================="
    )
    print(header)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Frame not received.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if len(faces) == 0:
                print(f"| {timestamp:<19} | {'No face detected':<20} | {'---':<13} | {'---':<10} |")
            else:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                roi = frame[y:y+h, x:x+w]
                emotion, confidence = predict_emotion(roi, model, label_mapper)
                face_str = f"({x},{y},{w},{h})"
                print(f"| {timestamp:<19} | {face_str:<20} | {emotion:<13} | {confidence:6.2f}%   |")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
    finally:
        cap.release()


if __name__ == "__main__":
    dummy = np.zeros((48, 48, 3), dtype=np.uint8)
    predict_emotion(dummy, model, mapper)
    webcam_emotion_cli(model, mapper, face_cascade)