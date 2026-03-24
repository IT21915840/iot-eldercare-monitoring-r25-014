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
import zlib
from typing import Dict, Any, Tuple, List, Optional
from sklearn.base import InconsistentVersionWarning

# --- Configuration and Initialization ---

# Filter warnings from sklearn and others
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module="sklearn")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Patient and Model Configuration
PATIENT_CONFIGS = [
    {"id": "P001", "ip": "172.20.10.3", "name": "Patient Constantine"}
]

# File Paths
EMOTION_MODEL_JSON = "Emotional/model.yaml"
EMOTION_MODEL_H5 = "Emotional/model.h5"
FALL_MODEL_H5 = "Fall_detection/keras_Model.h5"
FALL_LABELS_TXT = "Fall_detection/labels.txt"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Mapping for Emotion Model output
EMOTION_MAPPER = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
    4: "sadness", 5: "surprise", 6: "neutral"
}

# Type alias for shared state
SharedState = Dict[str, Dict[str, Any]]


# --- Model Loading Functions ---

def load_emotion_model() -> Optional[tf.keras.Model]:
    """Loads the emotion prediction model from JSON and H5 files."""
    try:
        with open(EMOTION_MODEL_JSON, "r") as f:
            model_json = f.read()

        # Custom object scope is often necessary for models saved with custom layers
        with tf.keras.utils.custom_object_scope({"Functional": tf.keras.Model}):
            model = tf.keras.models.model_from_json(model_json)

        model.load_weights(EMOTION_MODEL_H5)
        return model
    except Exception as e:
        # logging.error(f"Error loading emotion model: {e}")
        return None


def load_fall_model() -> Tuple[Optional[tf.keras.Model], Optional[List[str]]]:
    """Loads the fall detection model and its class labels."""
    try:
        model = tf.keras.models.load_model(FALL_MODEL_H5, compile=False)
        with open(FALL_LABELS_TXT, "r") as f:
            class_names = [ln.strip() for ln in f.readlines()]
        return model, class_names
    except Exception as e:
        # logging.error(f"Error loading fall model: {e}")
        return None, None


def load_sleep_model() -> Tuple[Any, Any]:
    """Loads the sleep stage prediction model and the scaler."""
    try:
        # Assuming joblib is used for scikit-learn models/pipelines
        model = joblib.load("Sleep/sleep_stage_model.joblib")
        scaler = joblib.load("Sleep/scaler.joblib")
        return model, scaler
    except Exception as e:
        # logging.error(f"Error loading sleep model: {e}")
        return None, None


def load_detectors() -> Tuple[cv2.CascadeClassifier, cv2.HOGDescriptor]:
    """Loads OpenCV's face cascade and HOG pedestrian detector."""
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return face_cascade, hog


# --- Prediction Functions ---

def predict_emotion(face_roi: np.ndarray, model: Optional[tf.keras.Model]) -> Tuple[str, float]:
    """
    Predicts emotion from a detected face region of interest (ROI).
    Returns (emotion_label, confidence_percentage).
    """
    try:
        if model is None:
            return "Unknown", 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype(np.float32) / 255.0
        # Convert to 3-channel (RGB) for model input
        rgb = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_final = np.expand_dims(rgb, axis=0)  # Add batch dimension

        preds = model.predict(img_final, verbose=0)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]) * 100.0)

        return EMOTION_MAPPER.get(idx, "Unknown"), conf

    except Exception as e:
        # logging.error(f"Emotion prediction error: {e}")
        return "Error", 0.0


def predict_fall(frame: np.ndarray, hog: cv2.HOGDescriptor, model: Optional[tf.keras.Model],
                 labels: Optional[List[str]]) -> Tuple[str, float]:
    """
    Detects a person using HOG and predicts fall status using the Keras model.
    Returns (fall_status_label, confidence_percentage).
    """
    try:
        # HOG detection for person
        (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        if len(boxes) == 0:
            return "No person", 0.0

        # Preprocessing for the Keras model
        resized = cv2.resize(frame, (224, 224))
        arr = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
        # Assuming the model uses a [-1, 1] normalization
        arr = (arr / 127.5) - 1.0

        if model is None or labels is None:
            return "Unknown", 0.0

        pred = model.predict(arr, verbose=0)
        idx = int(np.argmax(pred))

        label = labels[idx] if idx < len(labels) else "Unknown"
        conf = float(pred[0][idx] * 100.0)

        return label, conf

    except Exception as e:
        # logging.error(f"Fall prediction error: {e}")
        return "Error", 0.0


def predict_sleep_stage(model: Any, scaler: Any, spo2: int, hr: int, temp: float) -> str:
    """
    Predicts the sleep stage using physiological data.
    """
    try:
        if model is None or scaler is None:
            return "Error"

        # Prepare features for the scikit-learn model
        features = pd.DataFrame([[spo2, hr, temp]], columns=["spo2", "hr", "temp"])
        scaled = scaler.transform(features)

        pred = model.predict(scaled)
        return str(pred[0])  # Convert prediction to string

    except Exception as e:
        # logging.error(f"Sleep prediction error: {e}")
        return "Error"


# --- Criticality Assessment ---

def evaluate_criticality(hr: int, spo2: int, temp: float, stage: str, fall_status: str, emotion: str,
                         esp_critical_flag: int) -> Tuple[str, str]:
    """
    Evaluates patient status and assigns a criticality level and associated alerts.
    Returns (level, alert_string).
    """
    alerts: List[str] = []
    level: str = "Normal"

    # P0: Critical Thresholds (Highest Priority)
    if esp_critical_flag == 1:
        alerts.append("SENSOR CRITICAL THRESHOLD (HR/SpO2)")
        level = "Critical"

    if isinstance(fall_status, str) and ("fall" in fall_status.lower()):
        alerts.append("AI FALL DETECTED")
        level = "Critical"

    # P1: High Thresholds
    if hr < 40 or hr > 150:
        alerts.append("Extreme Heart Rate")
        if level != "Critical":
            level = "High"

    elif spo2 < 90 and spo2 >= 85:
        alerts.append("Warning: Low SpO2")
        if level != "Critical":
            level = "High"

    # P2: Moderate Thresholds
    if emotion in ["anger", "fear", "sadness"]:
        alerts.append(f"Negative Emotion: {emotion}")
        if level == "Normal":
            level = "Moderate"

    if stage == "Error":
        alerts.append("Sleep Stage Prediction Error")
        if level == "Normal":
            level = "Moderate"

    if hr < 50 or hr > 120:
        alerts.append("Warning: Abnormal Heart Rate")
        if level == "Normal":
            level = "Moderate"

    if temp < 29 or temp > 37:
        alerts.append("Abnormal Temperature")
        if level == "Normal":
            level = "Moderate"

    # Default/Stable Status
    if not alerts:
        alerts.append("Stable Condition")

    return level, ", ".join(alerts)


# --- Cloud Communication ---

async def send_to_cloud(payload: Dict[str, Any], priority: str = "P3_BUFFER") -> None:
    """
    Simulates sending data to a cloud endpoint with compression and priority logging.
    """
    json_data = json.dumps(payload).encode("utf-8")
    compressed_data = zlib.compress(json_data, level=9)

    # Log based on priority level
    patient_id = payload.get('patient_id', '?')
    data_size = len(compressed_data)

    if priority == "P0_INSTANT":
        logging.warning(f"[P0] {patient_id} - INSTANT CRITICAL ALERT - {data_size} bytes")
    elif priority == "P1_IMMEDIATE":
        logging.info(f"[P1] {patient_id} - IMMEDIATE HIGH ALERT - {data_size} bytes")
    elif priority == "P2_STANDARD":
        logging.info(f"[P2] {patient_id} - STANDARD MONITORING")
    else:
        logging.debug(f"[P3] {patient_id} - BUFFERED DATA")


# --- Asynchronous and Threaded Processes ---

def webcam_loop(shared_state: SharedState) -> None:
    """
    Dedicated thread for synchronous webcam operations (CV/ML prediction).
    Updates the shared_state dictionary with fall and emotion data.
    """
    # Load detectors and models once per thread
    face_cascade, hog = load_detectors()
    emotion_model = load_emotion_model()
    fall_model, fall_labels = load_fall_model()

    # Only processes the first patient for simplicity (P001)
    pid = PATIENT_CONFIGS[0]["id"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open webcam (cv2.VideoCapture(0)).")
        return

    fall_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Webcam frame read failed.")
            break

        # 1. Fall Detection
        fall_status, fall_conf = predict_fall(frame, hog, fall_model, fall_labels)

        # Fall persistence check (10 continuous fall frames)
        if "fall" in str(fall_status).lower():
            fall_counter += 1
        else:
            fall_counter = 0

        # Get or initialize the patient's state
        state = shared_state.get(pid, {})

        if fall_counter >= 10:
            state["fall_alert"] = True
            fall_status = "Continuous Fall"
        else:
            state["fall_alert"] = False

        # 2. Emotion Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            roi = frame[y: y + h, x: x + w]
            emotion, emo_conf = predict_emotion(roi, emotion_model)
            face_coords = f"({x},{y},{w},{h})"
        else:
            emotion, emo_conf, face_coords = "No face", 0.0, "---"

        # Update the shared state
        state["fall"] = (fall_status, fall_conf)
        state["emotion"] = (emotion, emo_conf)
        state["face"] = face_coords
        shared_state[pid] = state


async def receive_sensor_data(queue: asyncio.Queue, model: Any, scaler: Any, patient_id: str,
                              websocket_url: str) -> None:
    """
    Asynchronous task to connect to the sensor WebSocket and process incoming data.

    FIX: Changed 'timeout' to 'open_timeout' in websockets.connect()
    to resolve the 'unexpected keyword argument' error.
    """
    reconnect_delay = 5
    while True:
        try:
            logging.info(f"Connecting to {patient_id} sensor at {websocket_url}...")
            # ESTABLISH CONNECTION - THE FIX IS HERE
            async with websockets.connect(websocket_url, open_timeout=10) as ws:
                logging.info(f"Successfully connected to {patient_id} sensor.")

                # Listen for messages
                async for message in ws:
                    try:
                        data = json.loads(message)
                        if data.get("pid") != patient_id:
                            continue

                        # Extract and type-cast sensor data
                        hr = int(data.get("hr", 0))
                        spo2 = int(data.get("spo2", 0))
                        temp = float(data.get("temp", 0.0))
                        critical_flag = int(data.get("critical_flag", 0))

                        # Predict sleep stage (synchronous, but quick)
                        stage = predict_sleep_stage(model, scaler, spo2, hr, temp)

                        # Put combined data into the queue for unified processing
                        await queue.put({
                            "patient_id": patient_id,
                            "hr": hr,
                            "spo2": spo2,
                            "temp": temp,
                            "stage": stage,
                            "esp_critical_flag": critical_flag
                        })

                    except json.JSONDecodeError:
                        logging.warning(f"Received non-JSON message from {patient_id}.")
                    except Exception as e:
                        # logging.error(f"Error processing sensor message from {patient_id}: {e}")
                        continue

        except websockets.exceptions.ConnectionClosedOK:
            logging.info(f"{patient_id} connection closed normally. Reconnecting...")
        except websockets.exceptions.WebSocketException as e:
            logging.error(f"WebSocket error for {patient_id}: {e}. Retrying in {reconnect_delay}s.")
        except asyncio.TimeoutError:
            logging.warning(f"Connection attempt to {patient_id} timed out. Retrying in {reconnect_delay}s.")
        except Exception as e:
            # The original error handling was correct, but the use of the variable 'e'
            # inside the logging message is crucial for diagnosing unexpected errors.
            logging.error(f"Unexpected error for {patient_id} sensor: {e}. Retrying in {reconnect_delay}s.")

        await asyncio.sleep(reconnect_delay)


# --- Main Asynchronous Monitoring Loop ---

async def unified_monitor() -> None:
    """
    The main coordinator function. Initializes resources, starts tasks,
    and handles the main data processing and logging loop.
    """
    # 1. Initialize Shared State and Models
    shared_state: SharedState = {
        cfg["id"]: {"fall": ("---", 0.0), "emotion": ("---", 0.0), "face": "---", "fall_alert": False}
        for cfg in PATIENT_CONFIGS
    }
    sleep_model, sleep_scaler = load_sleep_model()

    if sleep_model is None or sleep_scaler is None:
        logging.error("Failed to load sleep model/scaler. Cannot run monitoring.")
        return

    # 2. Start Webcam Thread
    logging.info("Starting webcam processing thread...")
    threading.Thread(target=webcam_loop, args=(shared_state,), daemon=True).start()

    # 3. Start Sensor Data Receiving Tasks
    queue = asyncio.Queue()
    logging.info("Starting sensor data WebSocket tasks...")
    for cfg in PATIENT_CONFIGS:
        websocket_url = f"ws://{cfg['ip']}/ws"
        asyncio.create_task(
            receive_sensor_data(queue, sleep_model, sleep_scaler, cfg["id"], websocket_url)
        )

    # 4. Console Output Header
    header = (
        f"| {'TIME':<12} | {'PID':<5} | {'HR':<5} | {'SpO2':<6} | {'STAGE':<14} | "
        f"{'FALL STATUS':<15} | {'EMOTION':<12} | {'CRITICALITY LEVEL':<18} | "
        f"{'ALERT SOURCE':<35} |"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # 5. Main Processing Loop
    while True:
        # Wait for data from the sensor queue
        data: Dict[str, Any] = await queue.get()
        pid = data["patient_id"]
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Retrieve visual data from the shared state
        patient_state = shared_state.get(pid, shared_state[PATIENT_CONFIGS[0]["id"]])
        fall_status, _ = patient_state.get("fall", ("---", 0.0))
        emotion, _ = patient_state.get("emotion", ("---", 0.0))

        # 6. Evaluate Criticality
        level, alerts = evaluate_criticality(
            data["hr"], data["spo2"], data["temp"], data["stage"],
            fall_status, emotion, data["esp_critical_flag"]
        )

        # Override level/alerts if continuous fall is detected
        if patient_state.get("fall_alert"):
            level = "Critical"
            alerts = alerts + ", AI CONTINUOUS FALL"

        # 7. Determine Cloud Priority
        priority = "P3_BUFFER"
        if level == "Critical":
            priority = "P0_INSTANT"
        elif level == "High":
            priority = "P1_IMMEDIATE"
        elif level == "Moderate":
            priority = "P2_STANDARD"

        # 8. Send to Cloud (as a background task)
        # Include all relevant data in the payload for the cloud
        cloud_payload = {**data, "level": level, "alerts": alerts, "emotion": emotion, "fall_status": fall_status}
        asyncio.create_task(send_to_cloud(cloud_payload, priority=priority))

        # 9. Print to Console
        print(
            f"| {timestamp:<12} | {pid:<5} | {data['hr']:<5} | {data['spo2']:<6} | "
            f"{str(data['stage']):<14} | {fall_status:<15} | {emotion:<12} | "
            f"{level:<18} | {alerts:<35} |"
        )


# --- Execution ---

if __name__ == "__main__":
    try:
        asyncio.run(unified_monitor())
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring system stopped by user.")
    except Exception as main_e:
        logging.error(f"A fatal error occurred in the main asyncio loop: {main_e}")