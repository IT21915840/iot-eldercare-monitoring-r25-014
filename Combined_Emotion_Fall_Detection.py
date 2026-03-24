import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime

EMOTION_MODEL_JSON = "Emotional/model.yaml"
EMOTION_MODEL_H5 = "Emotional/model.h5"
EMOTION_MAPPER = {
    0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness',
    4: 'sadness', 5: 'surprise', 6: 'neutral'
}

FALL_MODEL_H5 = "Fall_detection/keras_Model.h5"
FALL_LABELS_TXT = "Fall_detection/labels.txt"

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

np.set_printoptions(suppress=True)


def load_detectors():
    print("Loading detectors...")

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError(f"Cannot load Haar cascade from {HAAR_CASCADE_PATH}")

    hog_detector = cv2.HOGDescriptor()
    hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("Haar Cascade and HOG Detector loaded successfully.")
    return face_cascade, hog_detector


def load_emotion_model():
    print("Loading emotion model...")
    try:
        with open(EMOTION_MODEL_JSON, "r") as json_file:
            model_json = json_file.read()
        with tf.keras.utils.custom_object_scope({'Functional': tf.keras.Model}):
            model = tf.keras.models.model_from_json(model_json)
        model.load_weights(EMOTION_MODEL_H5)
        print("Emotion model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return None


def load_fall_model():
    print("Loading fall detection model...")
    try:
        model = tf.keras.models.load_model(FALL_MODEL_H5, compile=False)
        with open(FALL_LABELS_TXT, "r") as f:
            class_names = f.readlines()
        print("Fall detection model loaded successfully.")
        return model, class_names
    except Exception as e:
        print(f"Error loading fall detection model: {e}")
        return None, None


def predict_emotion(face_roi, model):
    try:
        if len(face_roi.shape) == 3:
            img_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = face_roi

        img_resized = cv2.resize(img_gray, (48, 48))

        img_normalized = img_resized.astype(np.float32) / 255.0
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

        img_final = np.expand_dims(img_rgb, axis=0)

        predictions = model.predict(img_final, verbose=0)
        predicted_index = np.argmax(predictions[0])
        emotion = EMOTION_MAPPER.get(predicted_index, "Unknown")
        confidence = np.max(predictions[0]) * 100
        return emotion, confidence

    except Exception as e:
        return "Error", 0.0


def predict_fall_status(frame, hog_detector, model, class_names):
    (boxes, weights) = hog_detector.detectMultiScale(
        frame, winStride=(4, 4), padding=(8, 8), scale=1.05
    )

    if len(boxes) == 0:
        return "No person", 0.0

    try:
        image_for_model = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image_for_model = np.asarray(image_for_model, dtype=np.float32).reshape(1, 224, 224, 3)
        image_for_model = (image_for_model / 127.5) - 1

        prediction = model.predict(image_for_model, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index][2:].strip()
        confidence = prediction[0][index] * 100
        return class_name, confidence

    except Exception as e:
        return "Error", 0.0


def main():
    try:
        face_cascade, hog_detector = load_detectors()
        emotion_model = load_emotion_model()
        fall_model, fall_labels = load_fall_model()

        if not all([emotion_model, fall_model, fall_labels]):
            print("One or more models failed to load. Exiting.")
            return

    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("Warming up models...")
    dummy_frame_224 = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_roi_48 = np.zeros((48, 48, 3), dtype=np.uint8)
    predict_fall_status(dummy_frame_224, hog_detector, fall_model, fall_labels)
    predict_emotion(dummy_roi_48, emotion_model)
    print("Models warmed up. Starting webcam feed.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    header = (
        f"| {'Timestamp':<28} | {'Fall Status':<20} | {'Fall Conf.':<10} | "
        f"{'Emotion':<13} | {'Emotion Conf.':<13} | {'Face Coords (x,y,w,h)':<24} |"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Frame not received. Exiting...")
                break

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            fall_status, fall_conf = predict_fall_status(frame, hog_detector, fall_model, fall_labels)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                roi = frame[y:y + h, x:x + w]

                emotion, emotion_conf = predict_emotion(roi, emotion_model)
                face_str = f"({x},{y},{w},{h})"
                emotion_conf_str = f"{emotion_conf:6.2f}%"
            else:
                emotion, emotion_conf_str, face_str = "No face", "---", "---"

            fall_conf_str = f"{fall_conf:6.2f}%" if fall_conf > 0 else "---"

            print(
                f"| {timestamp_str:<28} | {fall_status:<20} | {fall_conf_str:<10} | "
                f"{emotion:<13} | {emotion_conf_str:<13} | {face_str:<24} |"
            )

            keyboard_input = cv2.waitKey(1)
            if keyboard_input == 27:
                print("\nDetection stopped by user.")
                break

    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
    finally:
        cap.release()
        print("Webcam released.")


if __name__ == "__main__":
    main()