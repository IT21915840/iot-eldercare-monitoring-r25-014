import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import logging
import datetime
import joblib
import pandas as pd
import warnings
import cv2
import numpy as np
import threading
import os
from typing import Dict, Any, Tuple, List, Optional
from sklearn.base import InconsistentVersionWarning
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import uvicorn
import aiomqtt
from fastapi.responses import StreamingResponse

# ── AI backend: TensorFlow (.h5) → tflite-runtime (.tflite) → disabled ───────
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

KERAS_AVAILABLE  = False
TFLITE_AVAILABLE = False
tf               = None
TFLiteInterpreter = None

try:
    import tensorflow as _tf
    tf = _tf
    KERAS_AVAILABLE  = True
    TFLITE_AVAILABLE = True
    TFLiteInterpreter = _tf.lite.Interpreter
    logging.info(f"TensorFlow {_tf.__version__} — Keras .h5 models supported.")
except Exception:
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter as _TFLite
        except ImportError:
            from tflite_runtime.interpreter import Interpreter as _TFLite
        TFLiteInterpreter = _TFLite
        TFLITE_AVAILABLE  = True
        logging.info("TFLite runtime — .tflite models supported.")
    except Exception:
        logging.warning("No AI runtime found. Vision features disabled. Please install 'ai-edge-litert-nightly'.")

warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ── Configuration ─────────────────────────────────────────────────────────────
MQTT_BROKER_HOST = "localhost"
MQTT_PORT        = 1883
MQTT_TOPIC_RAW   = "vitals/raw"
MQTT_TOPIC_PROCESSED = "vitals/processed"

PATIENT_CONFIGS = [
    {"id": "P001", "name": "Patient Constantine"}
]

# ── Model paths (.h5 preferred when TF available, .tflite as fallback) ────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMOTION_H5     = os.path.join(BASE_DIR, "Emotional", "model.h5")
EMOTION_TFLITE = os.path.join(BASE_DIR, "Emotional", "emotion_model.tflite")
FALL_H5        = os.path.join(BASE_DIR, "Fall_detection", "keras_Model.h5")
FALL_TFLITE    = os.path.join(BASE_DIR, "Fall_detection", "fall_model.tflite")
FALL_LABELS_TXT  = os.path.join(BASE_DIR, "Fall_detection", "labels.txt")
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

EMOTION_MAPPER = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
    4: "sadness", 5: "surprise", 6: "neutral"
}

SharedState = Dict[str, Dict[str, Any]]


# ── Model loader: tries Keras .h5 first, then .tflite ────────────────────────
def load_vision_model(h5_path: str, tflite_path: str):
    """Returns (model_object, mode) where mode is 'keras' or 'tflite'."""
    if KERAS_AVAILABLE and tf is not None:
        import os
        if os.path.exists(h5_path):
            try:
                class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
                    def __init__(self, **kwargs):
                        kwargs.pop('groups', None)
                        super().__init__(**kwargs)

                model = tf.keras.models.load_model(
                    h5_path, 
                    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, 
                    compile=False
                )
                logging.info(f"Keras model loaded: {h5_path}")
                return model, "keras"
            except Exception as e:
                logging.warning(f"Keras load failed for {h5_path}: {e}")
    if TFLITE_AVAILABLE and TFLiteInterpreter is not None:
        import os
        if os.path.exists(tflite_path):
            try:
                interp = TFLiteInterpreter(model_path=tflite_path)
                interp.allocate_tensors()
                logging.info(f"TFLite model loaded: {tflite_path}")
                return interp, "tflite"
            except Exception as e:
                logging.warning(f"TFLite load failed for {tflite_path}: {e}")
    logging.warning(f"No model available at {h5_path} or {tflite_path}")
    return None, None


def run_vision_inference(model, mode: str, input_arr: np.ndarray) -> np.ndarray:
    """Unified inference for either Keras or TFLite model."""
    if model is None or mode is None:
        return np.array([])
    if mode == "keras":
        return model.predict(input_arr, verbose=0)
    else:
        input_details  = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], input_arr)
        model.invoke()
        return model.get_tensor(output_details[0]['index'])



def load_sleep_model():
    try:
        model_path = os.path.join(BASE_DIR, "Sleep", "sleep_stage_model.joblib")
        scaler_path = os.path.join(BASE_DIR, "Sleep", "scaler.joblib")
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info("Sleep model loaded.")
        return model, scaler
    except Exception as e:
        logging.error(f"Sleep model load failed: {e}")
        return None, None


def load_fall_labels() -> Optional[List[str]]:
    try:
        with open(FALL_LABELS_TXT, "r") as f:
            return [ln.strip() for ln in f.readlines()]
    except Exception as e:
        logging.error(f"Fall labels load failed: {e}")
        return None


def load_detectors():
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return face_cascade, hog


# ── AI inference ─────────────────────────────────────────────────────────────
def predict_emotion(face_roi: np.ndarray, model, mode: str) -> Tuple[str, float]:
    try:
        if model is None or face_roi.size == 0:
            return "neutral", 0.0
        gray       = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized    = cv2.resize(gray, (48, 48))
        rgb        = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        img        = rgb.astype(np.float32) / 255.0
        img_batch  = np.expand_dims(img, axis=0)
        preds      = run_vision_inference(model, mode, img_batch)
        if preds.size == 0: return "neutral", 0.0
        idx        = int(np.argmax(preds[0]))
        return EMOTION_MAPPER.get(idx, "neutral"), float(np.max(preds[0]) * 100)
    except Exception as e:
        logging.warning(f"Emotion inference error: {e}")
        return "neutral", 0.0


def predict_fall(frame: np.ndarray, hog, model, mode: str, labels) -> Tuple[str, float]:
    try:
        (boxes, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        if len(boxes) == 0:
            return "No person", 0.0
        if model is None or labels is None:
            return "Unknown", 0.0
        resized = cv2.resize(frame, (224, 224)).astype(np.float32)
        arr     = ((resized / 127.5) - 1.0).reshape(1, 224, 224, 3)
        pred    = run_vision_inference(model, mode, arr)
        if pred.size == 0: return "No person", 0.0
        idx     = int(np.argmax(pred[0]))
        raw_label = labels[idx] if idx < len(labels) else "Unknown"
        
        # Clean the label to prevent "Not Fallen" from triggering substring checks for "Fall"
        raw_upper = raw_label.upper()
        if "NOT" in raw_upper or "NO" in raw_upper:
            final_label = "Normal"
        elif "FALL" in raw_upper:
            final_label = "A FALL HAS OCCURED"
        else:
            final_label = raw_label
            
        return final_label, float(pred[0][idx] * 100)
    except Exception as e:
        logging.warning(f"Fall inference error: {e}")
        return "No person", 0.0


def predict_sleep_stage(model, scaler, spo2: int, hr: int, temp: float) -> str:
    try:
        if model is None or scaler is None:
            return "Awake"
        features = pd.DataFrame([[spo2, hr, temp]], columns=["spo2", "hr", "temp"])
        return str(model.predict(scaler.transform(features))[0])
    except Exception:
        return "Awake"


def evaluate_criticality(hr, spo2, temp, stage, fall_status, emotion, esp_crit) -> Tuple[str, str]:
    alerts, level = [], "Normal"
    if esp_crit == 1:
        alerts.append("SENSOR CRITICAL THRESHOLD"); level = "Critical"
    if isinstance(fall_status, str) and "fall" in fall_status.lower():
        alerts.append("AI FALL DETECTED"); level = "Critical"
    if hr > 100:
        alerts.append(f"Abnormal Heart Rate: Too High ({hr} BPM)")
        if level != "Critical": level = "High"
    elif hr < 55:
        alerts.append(f"Abnormal Heart Rate: Too Low ({hr} BPM)")
        if level != "Critical": level = "High"
        
    if spo2 < 90:
        alerts.append(f"Warning: Low SpO2 ({spo2}%)")
        if level != "Critical": level = "High"
    if emotion in ["anger", "fear", "sadness"] and level == "Normal":
        alerts.append(f"Negative Emotion: {emotion}"); level = "Moderate"
    if temp < 29 or temp > 37:
        alerts.append(f"Abnormal Temperature ({temp}°C)")
        if level == "Normal": level = "Moderate"
    if not alerts:
        alerts.append("Stable Condition")
    return level, ", ".join(alerts)


# ── WebSocket manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        try: self.active_connections.remove(ws)
        except ValueError: pass

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active_connections:
            try: await ws.send_text(message)
            except Exception: dead.append(ws)
        for ws in dead: self.disconnect(ws)


manager = ConnectionManager()

shared_state: SharedState = {
    cfg["id"]: {"fall": ("No person", 0.0), "emotion": ("neutral", 0.0), "fall_alert": False}
    for cfg in PATIENT_CONFIGS
}

# ── Global Frame Buffer for Streaming ──────────────────────────────────────────
current_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()
CAMERA_ACTIVE = False

def get_drawing_frame():
    global current_frame
    with frame_lock:
        if current_frame is None:
            return None
        return current_frame.copy()

async def generate_mjpeg():
    # Wait for the first frame to populate to prevent hanging the initial HTTP headers
    timeout = 0
    while get_drawing_frame() is None:
        await asyncio.sleep(0.1)
        timeout += 1
        if timeout > 50: # 5 seconds
            break

    while True:
        if not CAMERA_ACTIVE:
            break
        frame = get_drawing_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.1) # ~10 FPS is plenty for MJPEG monitoring


# ── Camera thread ─────────────────────────────────────────────────────────────
def webcam_loop(emotion_model, emotion_mode, fall_model, fall_mode, fall_labels) -> None:
    global CAMERA_ACTIVE
    face_cascade, hog = load_detectors()
    pid = PATIENT_CONFIGS[0]["id"]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.warning("No camera found — vision AI disabled.")
        CAMERA_ACTIVE = False
        return
    
    CAMERA_ACTIVE = True

    fall_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        fall_status, fall_conf = predict_fall(frame, hog, fall_model, fall_mode, fall_labels)
        fall_counter = (fall_counter + 1) if "fall" in str(fall_status).lower() else 0

        state = shared_state.get(pid, {})
        state["fall_alert"] = fall_counter >= 10
        if state["fall_alert"]:
            fall_status = "Continuous Fall"

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            emotion, emo_conf = predict_emotion(frame[y:y+h, x:x+w], emotion_model, emotion_mode)
        else:
            emotion, emo_conf = "No face", 0.0

        state["fall"]    = (fall_status, fall_conf)
        state["emotion"] = (emotion, emo_conf)
        shared_state[pid] = state

        # Update global frame for streaming
        global current_frame
        with frame_lock:
            current_frame = frame.copy()


# ── MQTT listener ─────────────────────────────────────────────────────────────
async def mqtt_listener(sleep_model, sleep_scaler) -> None:
    reconnect_delay = 5
    while True:
        try:
            logging.info(f"[MQTT] Connecting to {MQTT_BROKER_HOST}:{MQTT_PORT} ...")
            async with aiomqtt.Client(hostname=MQTT_BROKER_HOST, port=MQTT_PORT) as client:
                await client.subscribe(MQTT_TOPIC_RAW)
                logging.info(f"[MQTT] Subscribed to '{MQTT_TOPIC_RAW}'")
                async for message in client.messages:
                    try:
                        raw  = json.loads(message.payload)
                        pid  = str(raw.get("pid", PATIENT_CONFIGS[0]["id"]))
                        hr   = int(raw.get("hr",   0))
                        spo2 = int(raw.get("spo2", 0))
                        temp = float(raw.get("temp", 0.0))
                        crit = int(raw.get("crit",  0))

                        # Hybrid Logic: Use provided stage (e.g. from Mock/Sensor) if available, otherwise predict via AI
                        stage = raw.get("stage")
                        if not stage or stage == "Warmup":
                            stage = predict_sleep_stage(sleep_model, sleep_scaler, spo2, hr, temp)
                        patient_state = shared_state.get(pid, shared_state[PATIENT_CONFIGS[0]["id"]])
                        fall_status, fall_conf = patient_state.get("fall", ("No person", 0.0))
                        emotion, emo_conf      = patient_state.get("emotion", ("neutral", 0.0))
                        if patient_state.get("fall_alert"):
                            fall_status = "Continuous Fall"

                        level, alerts = evaluate_criticality(hr, spo2, temp, stage, fall_status, emotion, crit)
                        patient_name  = next((c["name"] for c in PATIENT_CONFIGS if c["id"] == pid), f"Patient {pid}")

                        payload = {
                            "patient_id":         pid,
                            "patient_name":       patient_name,
                            "hr":                 hr,
                            "spo2":               spo2,
                            "temp":               temp,
                            "stage":              stage,
                            "level":              level,
                            "alerts":             alerts,
                            "emotion":            emotion,
                            "fall_status":        fall_status,
                            "emotion_confidence": emo_conf,
                            "fall_confidence":    fall_conf,
                            "timestamp":          datetime.datetime.now().strftime("%H:%M:%S"),
                        }

                        payload_str = json.dumps(payload)
                        await manager.broadcast(payload_str)
                        await client.publish(MQTT_TOPIC_PROCESSED, payload=payload_str)
                        logging.info(f"[BROADCAST & PUBLISH] {pid} HR:{hr} SpO2:{spo2} Temp:{temp} → {level}")

                    except Exception as e:
                        logging.warning(f"[MQTT] Processing error: {e}")

        except aiomqtt.MqttError as e:
            logging.error(f"[MQTT] Broker error: {e}. Retrying in {reconnect_delay}s ...")
        await asyncio.sleep(reconnect_delay)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/video_feed")
async def video_feed():
    if not CAMERA_ACTIVE:
        raise HTTPException(status_code=503, detail="Camera hardware offline.")
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logging.info("Downstream client connected.")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    sleep_model, sleep_scaler   = load_sleep_model()
    emotion_model, emotion_mode = load_vision_model(EMOTION_H5, EMOTION_TFLITE)
    fall_model, fall_mode       = load_vision_model(FALL_H5, FALL_TFLITE)
    fall_labels                 = load_fall_labels()

    logging.info("Starting camera thread ...")
    threading.Thread(
        target=webcam_loop,
        args=(emotion_model, emotion_mode, fall_model, fall_mode, fall_labels),
        daemon=True
    ).start()

    logging.info("Starting MQTT listener ...")
    asyncio.create_task(mqtt_listener(sleep_model, sleep_scaler))


if __name__ == "__main__":
    if sys.platform == 'win32':
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()

    config = uvicorn.Config(app, host="0.0.0.0", port=8001, loop="none")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
