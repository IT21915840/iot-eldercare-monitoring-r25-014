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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
PATIENT_CONFIGS = [
    {"id": "P001", "ip": "172.20.10.3", "name": "Patient Fernando"}
]
EMOTION_MODEL_JSON = "Emotional/model.yaml"
EMOTION_MODEL_H5 = "Emotional/model.h5"
FALL_MODEL_H5 = "Fall_detection/keras_Model.h5"
FALL_LABELS_TXT = "Fall_detection/labels.txt"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EMOTION_MAPPER = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
    4: "sadness", 5: "surprise", 6: "neutral"
}
SharedState = Dict[str, Dict[str, Any]]
def load_emotion_model() -> Optional[tf.keras.Model]:
    try:
        with open(EMOTION_MODEL_JSON, "r") as f:
            model_json = f.read()
        with tf.keras.utils.custom_object_scope({"Functional": tf.keras.Model}):
            model = tf.keras.models.model_from_json(model_json)
        model.load_weights(EMOTION_MODEL_H5)
        return model
    except Exception as e:
        return None
def load_fall_model() -> Tuple[Optional[tf.keras.Model], Optional[List[str]]]:
    try:
        model = tf.keras.models.load_model(FALL_MODEL_H5, compile=False)
        with open(FALL_LABELS_TXT, "r") as f:
            class_names = [ln.strip() for ln in f.readlines()]
        return model, class_names
    except Exception as e:
        return None, None
def load_sleep_model() -> Tuple[Any, Any]:
    try:
        model = joblib.load("Sleep/sleep_stage_model.joblib")
        scaler = joblib.load("Sleep/scaler.joblib")
        return model, scaler
    except Exception as e:
        return None, None
def load_detectors() -> Tuple[cv2.CascadeClassifier, cv2.HOGDescriptor]:
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return face_cascade, hog
def predict_emotion(face_roi: np.ndarray, model: Optional[tf.keras.Model]) -> Tuple[str, float]:
    try:
        if model is None:
            return "Unknown", 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype(np.float32) / 255.0
        rgb = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_final = np.expand_dims(rgb, axis=0)
        preds = model.predict(img_final, verbose=0)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]) * 100.0)
        return EMOTION_MAPPER.get(idx, "Unknown"), conf
    except Exception as e:
        return "Error", 0.0
def predict_fall(frame: np.ndarray, hog: cv2.HOGDescriptor, model: Optional[tf.keras.Model],
                 labels: Optional[List[str]]) -> Tuple[str, float]:
    try:
        (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        if len(boxes) == 0:
            return "No person", 0.0
        resized = cv2.resize(frame, (224, 224))
        arr = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
        arr = (arr / 127.5) - 1.0
        if model is None or labels is None:
            return "Unknown", 0.0
        pred = model.predict(arr, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx] if idx < len(labels) else "Unknown"
        conf = float(pred[0][idx] * 100.0)
        return label, conf
    except Exception as e:
        return "Error", 0.0
def predict_sleep_stage(model: Any, scaler: Any, spo2: int, hr: int, temp: float) -> str:
    try:
        if model is None or scaler is None:
            return "Error"
        features = pd.DataFrame([[spo2, hr, temp]], columns=["spo2", "hr", "temp"])
        scaled = scaler.transform(features)
        pred = model.predict(scaled)
        return str(pred[0])
    except Exception as e:
        return "Error"
def evaluate_criticality(hr: int, spo2: int, temp: float, stage: str, fall_status: str, emotion: str,
                         esp_critical_flag: int) -> Tuple[str, str]:
    alerts: List[str] = []
    level: str = "Normal"
    if esp_critical_flag == 1:
        alerts.append("SENSOR CRITICAL THRESHOLD (HR/SpO2)")
        level = "Critical"
    if isinstance(fall_status, str) and ("fall" in fall_status.lower()):
        alerts.append("AI FALL DETECTED")
        level = "Critical"
    if hr < 40 or hr > 150:
        alerts.append("Extreme Heart Rate")
        if level != "Critical": level = "High"
    elif spo2 < 90 and spo2 >= 85:
        alerts.append("Warning: Low SpO2")
        if level != "Critical": level = "High"
    if emotion in ["anger", "fear", "sadness"]:
        alerts.append(f"Negative Emotion: {emotion}")
        if level == "Normal": level = "Moderate"
    if stage == "Error":
        alerts.append("Sleep Stage Prediction Error")
        if level == "Normal": level = "Moderate"
    if hr < 50 or hr > 120:
        alerts.append("Warning: Abnormal Heart Rate")
        if level == "Normal": level = "Moderate"
    if temp < 29 or temp > 37:
        alerts.append("Abnormal Temperature")
        if level == "Normal": level = "Moderate"
    if not alerts:
        alerts.append("Stable Condition")
    return level, ", ".join(alerts)
async def send_to_cloud(payload: Dict[str, Any], priority: str = "P3_BUFFER") -> None:
    json_data = json.dumps(payload).encode("utf-8")
    compressed_data = zlib.compress(json_data, level=9)
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
def webcam_loop(shared_state: SharedState) -> None:
    face_cascade, hog = load_detectors()
    emotion_model = load_emotion_model()
    fall_model, fall_labels = load_fall_model()
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
        fall_status, fall_conf = predict_fall(frame, hog, fall_model, fall_labels)
        if "fall" in str(fall_status).lower():
            fall_counter += 1
        else:
            fall_counter = 0
        state = shared_state.get(pid, {})
        if fall_counter >= 10:
            state["fall_alert"] = True
            fall_status = "Continuous Fall"
        else:
            state["fall_alert"] = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            roi = frame[y: y + h, x: x + w]
            emotion, emo_conf = predict_emotion(roi, emotion_model)
            face_coords = f"({x},{y},{w},{h})"
        else:
            emotion, emo_conf, face_coords = "No face", 0.0, "---"
        state["fall"] = (fall_status, fall_conf)
        state["emotion"] = (emotion, emo_conf)
        state["face"] = face_coords
        shared_state[pid] = state
async def receive_sensor_data(queue: asyncio.Queue, model: Any, scaler: Any, patient_id: str,
                              websocket_url: str) -> None:
    reconnect_delay = 5
    while True:
        try:
            logging.info(f"Connecting to {patient_id} sensor at {websocket_url}...")
            async with websockets.connect(websocket_url, open_timeout=10) as ws:
                logging.info(f"Successfully connected to {patient_id} sensor.")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        if data.get("pid") != patient_id:
                            continue
                        hr = int(data.get("hr", 0))
                        spo2 = int(data.get("spo2", 0))
                        temp = float(data.get("temp", 0.0))
                        critical_flag = int(data.get("critical_flag", 0))
                        stage = predict_sleep_stage(model, scaler, spo2, hr, temp)
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
                        continue
        except Exception as e:
            logging.error(f"WebSocket error for {patient_id}: {e}. Retrying in {reconnect_delay}s.")
        await asyncio.sleep(reconnect_delay)
app = FastAPI()
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
manager = ConnectionManager()
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logging.info("Dashboard client disconnected")
@app.get("/")
async def get_homepage():
    return FileResponse("HTML/index.html")
async def unified_monitor() -> None:
    shared_state: SharedState = {
        cfg["id"]: {"fall": ("---", 0.0), "emotion": ("---", 0.0), "face": "---", "fall_alert": False}
        for cfg in PATIENT_CONFIGS
    }
    sleep_model, sleep_scaler = load_sleep_model()
    if sleep_model is None or sleep_scaler is None:
        logging.error("Failed to load sleep model/scaler. Cannot run monitoring.")
        return
    logging.info("Starting webcam processing thread...")
    threading.Thread(target=webcam_loop, args=(shared_state,), daemon=True).start()
    queue = asyncio.Queue()
    logging.info("Starting sensor data WebSocket tasks...")
    for cfg in PATIENT_CONFIGS:
        websocket_url = f"ws://{cfg['ip']}/ws"
        asyncio.create_task(
            receive_sensor_data(queue, sleep_model, sleep_scaler, cfg["id"], websocket_url)
        )
    logging.info("--- Console table output disabled. Monitoring started. ---")
    while True:
        data: Dict[str, Any] = await queue.get()
        pid = data["patient_id"]
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        patient_state = shared_state.get(pid, shared_state[PATIENT_CONFIGS[0]["id"]])
        fall_status, _ = patient_state.get("fall", ("---", 0.0))
        emotion, _ = patient_state.get("emotion", ("---", 0.0))
        level, alerts = evaluate_criticality(
            data["hr"], data["spo2"], data["temp"], data["stage"],
            fall_status, emotion, data["esp_critical_flag"]
        )
        if patient_state.get("fall_alert"):
            level = "Critical"
            alerts = alerts + ", AI CONTINUOUS FALL"
        priority = "P3_BUFFER"
        if level == "Critical":
            priority = "P0_INSTANT"
        elif level == "High":
            priority = "P1_IMMEDIATE"
        elif level == "Moderate":
            priority = "P2_STANDARD"
        fused_payload = {
            **data,
            "timestamp": timestamp,
            "level": level,
            "alerts": alerts,
            "emotion": emotion,
            "fall_status": fall_status,
            "patient_name": PATIENT_CONFIGS[0]["name"]
        }
        asyncio.create_task(send_to_cloud(fused_payload.copy(), priority=priority))
        try:
            await manager.broadcast(json.dumps(fused_payload))
        except Exception as e:
            logging.warning(f"Error broadcasting to WebSocket clients: {e}")
@app.on_event("startup")
async def startup_event():
    logging.info("FastAPI server starting up...")
    asyncio.create_task(unified_monitor())
if __name__ == "__main__":
    print("Starting Patient Monitoring Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)