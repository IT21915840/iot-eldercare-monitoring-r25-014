import asyncio
import websockets
import json
import logging
import datetime
import joblib
import pandas as pd
import warnings
import cv2
import numpy as np
import tensorflow as tf
import threading
from sklearn.base import InconsistentVersionWarning
warnings.filterwarnings("ignore", module="sklearn")

ESP32_IP = "172.20.10.3"
WEBSOCKET_URL = f"ws://{ESP32_IP}/ws"

EMOTION_MODEL_JSON = "Emotional/model.yaml"
EMOTION_MODEL_H5 = "Emotional/model.h5"
FALL_MODEL_H5 = "Fall_detection/keras_Model.h5"
FALL_LABELS_TXT = "Fall_detection/labels.txt"

EMOTION_MAPPER = {
    0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness',
    4: 'sadness', 5: 'surprise', 6: 'neutral'
}

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module="sklearn")

def load_emotion_model():
    try:
        with open(EMOTION_MODEL_JSON, "r") as json_file:
            model_json = json_file.read()
        with tf.keras.utils.custom_object_scope({'Functional': tf.keras.Model}):
            model = tf.keras.models.model_from_json(model_json)
        model.load_weights(EMOTION_MODEL_H5)
        print("[INFO] Emotion model loaded.")
        return model
    except Exception as e:
        print(f"[ERROR] Emotion model error: {e}")
        return None

def load_fall_model():
    try:
        model = tf.keras.models.load_model(FALL_MODEL_H5, compile=False)
        with open(FALL_LABELS_TXT, "r") as f:
            class_names = f.readlines()
        print("[INFO] Fall detection model loaded.")
        return model, class_names
    except Exception as e:
        print(f"[ERROR] Fall model error: {e}")
        return None, None

def load_sleep_model():
    try:
        model = joblib.load('Sleep/sleep_stage_model.joblib')
        scaler = joblib.load('Sleep/scaler.joblib')
        print("[INFO] Sleep model loaded.")
        return model, scaler
    except Exception as e:
        print(f"[ERROR] Sleep model error: {e}")
        return None, None

def load_detectors():
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return face_cascade, hog

def predict_emotion(face_roi, model):
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype(np.float32) / 255.0
        rgb = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_final = np.expand_dims(rgb, axis=0)
        preds = model.predict(img_final, verbose=0)
        idx = np.argmax(preds[0])
        return EMOTION_MAPPER.get(idx, "Unknown"), float(np.max(preds[0]) * 100)
    except Exception:
        return "No face", 0.0

def predict_fall(frame, hog, model, labels):
    (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    if len(boxes) == 0:
        return "No person", 0.0
    try:
        resized = cv2.resize(frame, (224, 224))
        arr = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
        arr = (arr / 127.5) - 1
        pred = model.predict(arr, verbose=0)
        idx = np.argmax(pred)
        label = labels[idx][2:].strip()
        return label, float(pred[0][idx] * 100)
    except Exception:
        return "Error", 0.0

def predict_sleep_stage(model, scaler, spo2, hr, temp):
    try:
        features = pd.DataFrame([[spo2, hr, temp]], columns=['spo2', 'hr', 'temp'])
        scaled = scaler.transform(features)
        pred = model.predict(scaled)
        return pred[0]
    except Exception:
        return "Error"

def evaluate_criticality(hr, spo2, temp, stage, fall_status, emotion):
    alerts = []
    level = "Normal"

    if hr == 0 and spo2 == 0:
        alerts.append("Wearable Sensor Disconnection (HR & SpO₂) - Check Device/Connection")
        return "Critical", ", ".join(alerts)

    if temp < 29 or temp > 37:
        alerts.append("Abnormal Temperature")
        level = "Moderate" if 27 <= temp <= 39 else "Critical"

    if hr < 50 or hr > 120:
        alerts.append("Abnormal Heart Rate")
        level = "High" if hr < 40 or hr > 150 else "Moderate"

    if spo2 < 94:
        alerts.append("Low SpO₂")
        level = "Critical" if spo2 < 85 else "High"

    if stage == "Error":
        alerts.append("Sleep Stage Prediction Error")
        level = "Moderate"

    if emotion in ["anger", "fear", "sadness"]:
        alerts.append(f"Negative Emotion: {emotion}")
        if level == "Normal":
            level = "Moderate"

    if fall_status.lower() == "falling" or fall_status.lower() == "fall detected":
        alerts.append("Fall Detected")
        level = "Critical"

    if not alerts:
        alerts.append("Stable Condition")

    return level, ", ".join(alerts)

def webcam_loop(shared_state):
    face_cascade, hog = load_detectors()
    emotion_model = load_emotion_model()
    fall_model, fall_labels = load_fall_model()
    fall_counter = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Webcam not accessible. Check device connection.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fall_status, fall_conf = predict_fall(frame, hog, fall_model, fall_labels)

        if "fall" in fall_status.lower():
            fall_counter += 1
        else:
            fall_counter = 0

        if fall_counter >= 10:
            shared_state['fall_alert'] = True
            fall_status = "Continuous Fall"
        else:
            shared_state['fall_alert'] = False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            roi = frame[y:y+h, x:x+w]
            emotion, emo_conf = predict_emotion(roi, emotion_model)
            face_coords = f"({x},{y},{w},{h})"
        else:
            emotion, emo_conf, face_coords = "No face", 0.0, "---"

        shared_state['fall'] = (fall_status, fall_conf)
        shared_state['emotion'] = (emotion, emo_conf)
        shared_state['face'] = face_coords

async def receive_sensor_data(queue, model, scaler):
    while True:
        try:
            async with websockets.connect(WEBSOCKET_URL) as ws:
                print("[INFO] Connected to ESP32 WebSocket.")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        hr = data.get('hr', 0)
                        spo2 = data.get('spo2', 0)
                        temp = data.get('temp', 0.0)
                        stage = predict_sleep_stage(model, scaler, spo2, hr, temp)
                        await queue.put({'hr': hr, 'spo2': spo2, 'temp': temp, 'stage': stage})
                    except Exception as e:
                        print(f"[ERROR] Data processing error: {e}")
        except Exception:
            print("[WARN] Reconnecting to ESP32 in 5s...")
            await asyncio.sleep(5)

async def unified_monitor():
    shared_state = {'fall': ("---", 0.0), 'emotion': ("---", 0.0),
                    'face': "---", 'fall_alert': False}

    sleep_model, sleep_scaler = load_sleep_model()
    if not all([sleep_model, sleep_scaler]):
        print("[ERROR] Sleep model missing. Exiting monitor.")
        return

    threading.Thread(target=webcam_loop, args=(shared_state,), daemon=True).start()
    queue = asyncio.Queue()
    asyncio.create_task(receive_sensor_data(queue, sleep_model, sleep_scaler))

    header = (
        f"| {'TIME':<23} | {'HR':<5} | {'SpO2':<6} | {'TEMP(°C)':<9} | {'SLEEP STAGE':<14} | "
        f"{'FALL STATUS':<15} | {'FALL CONF(%)':<13} | {'EMOTION':<12} | {'EMO CONF(%)':<11} | "
        f"{'CRITICALITY LEVEL':<18} | {'ALERT SOURCE':<35} |"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    while True:
        data = await queue.get()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        fall_status, fall_conf = shared_state['fall']
        emotion, emo_conf = shared_state['emotion']

        level, alerts = evaluate_criticality(
            data['hr'], data['spo2'], data['temp'], data['stage'], fall_status, emotion
        )

        if shared_state.get('fall_alert'):
            level = "Critical"
            alerts += ", Continuous Fall Detected"

        print(
            f"| {timestamp:<23} | {data['hr']:<5} | {data['spo2']:<6} | {data['temp']:<9.2f} | "
            f"{data['stage']:<14} | {fall_status:<15} | {fall_conf:<13.2f} | {emotion:<12} | {emo_conf:<11.2f} | "
            f"{level:<18} | {alerts:<35} |"
        )

if __name__ == "__main__":
    try:
        asyncio.run(unified_monitor())
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped by user.")