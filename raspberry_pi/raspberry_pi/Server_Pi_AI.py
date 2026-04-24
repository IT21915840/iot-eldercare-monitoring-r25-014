import asyncio
import json
import logging
import datetime
import joblib
import pandas as pd
import warnings
import cv2
import numpy as np
import threading
import time
from typing import Dict, Any, Tuple, List, Optional
from sklearn.base import InconsistentVersionWarning
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import aiomqtt
from fastapi.responses import StreamingResponse
import os
import sys

def resolve_path(path: str) -> str:
    """Helper to find files in current or parent directory."""
    if os.path.exists(path):
        return path
    parent_path = os.path.join("..", path)
    if os.path.exists(parent_path):
        return parent_path
    return path # Return original if not found to let loader handle missing file error

# ── AI backend: TensorFlow (.h5) → tflite-runtime (.tflite) → disabled ───────
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

PATIENT_CONFIGS = [
    {"id": "P001", "name": "Patient Constantine"}
]

# ── Model paths (.h5 preferred when TF available, .tflite as fallback) ────────
EMOTION_H5     = "Emotional/model.h5"
EMOTION_TFLITE = "Emotional/emotion_model.tflite"
FALL_H5        = "Fall_detection/keras_Model.h5"
FALL_TFLITE    = "Fall_detection/fall_model.tflite"
FALL_LABELS_TXT  = "Fall_detection/labels.txt"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CAPTURES_DIR    = "captures"

# Ensure captures directory exists
if not os.path.exists(CAPTURES_DIR):
    os.makedirs(CAPTURES_DIR)
    logging.info(f"Created captures directory: {CAPTURES_DIR}")

EMOTION_MAPPER = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
    4: "sadness", 5: "surprise", 6: "neutral"
}

SharedState = Dict[str, Dict[str, Any]]


# ── Model loader: tries Keras .h5 first, then .tflite ────────────────────────
def load_vision_model(h5_path: str, tflite_path: str):
    """Returns (model_object, mode) where mode is 'keras' or 'tflite'."""
    h5_res = resolve_path(h5_path)
    tf_res = resolve_path(tflite_path)

    if KERAS_AVAILABLE and tf is not None:
        if os.path.exists(h5_res):
            try:
                model = tf.keras.models.load_model(h5_res, compile=False)
                logging.info(f"Keras model loaded: {h5_res}")
                return model, "keras"
            except Exception as e:
                logging.warning(f"Keras load failed for {h5_res}: {e}")
    
    if TFLITE_AVAILABLE and TFLiteInterpreter is not None:
        if os.path.exists(tf_res):
            try:
                interp = TFLiteInterpreter(model_path=tf_res)
                interp.allocate_tensors()
                logging.info(f"TFLite model loaded: {tf_res}")
                return interp, "tflite"
            except Exception as e:
                logging.warning(f"TFLite load failed for {tf_res}: {e}")
    
    logging.warning(f"No model available at {h5_res} or {tf_res}")
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
        model_path = resolve_path("Sleep/sleep_stage_model.joblib")
        scaler_path = resolve_path("Sleep/scaler.joblib")
        if not os.path.exists(model_path): return None, None
        
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info(f"Sleep models loaded from {os.path.dirname(model_path)}")
        return model, scaler
    except Exception as e:
        logging.error(f"Sleep model load failed: {e}")
        return None, None


def load_fall_labels() -> Optional[List[str]]:
    try:
        path = resolve_path(FALL_LABELS_TXT)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            lines = f.readlines()
        
        parsed_labels = []
        for line in lines:
            line = line.strip()
            if not line: continue
            # Handle "0 Fallen" or "Fallen"
            parts = line.split(' ', 1)
            if len(parts) > 1 and parts[0].isdigit():
                parsed_labels.append(parts[1].strip())
            else:
                parsed_labels.append(line)
        return parsed_labels
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
        if model is None or labels is None:
            return "Unknown", 0.0
        
        # HOG check removed because it often fails to detect a 'person' when they are lying on the floor.
        # This was likely preventing the AI model from even running.
        # (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        # if len(boxes) == 0:
        #     return "No person", 0.0

        # Preprocessing: Use BGR
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Normalization: [-1, 1]
        arr     = ((resized / 127.5) - 1.0).reshape(1, 224, 224, 3)
        
        pred    = run_vision_inference(model, mode, arr)
        if pred.size == 0: return "No person", 0.0
        
        idx     = int(np.argmax(pred[0]))
        confidence = float(pred[0][idx] * 100)
        label = labels[idx] if idx < len(labels) else "Unknown"
        
        # Logging for identifying accuracy issues
        logging.info(f"[AI] Model Prediction: {label} ({confidence:.1f}%)")
            
        return label, confidence
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
    # Determine if sensors are actually connected (ignoring all zeros)
    is_disconnected = (hr == 0 and spo2 == 0 and temp == 0)
    
    if is_disconnected:
        return "Normal", "SENSOR DISCONNECTED"

    alerts, level = [], "Normal"
    if esp_crit == 1:
        alerts.append("SENSOR CRITICAL THRESHOLD"); level = "Critical"
    if isinstance(fall_status, str) and fall_status.lower().strip() == "fallen":
        alerts.append("AI FALL DETECTED"); level = "Critical"
    if hr < 40 or hr > 150:
        alerts.append("Extreme Heart Rate")
        if level != "Critical": level = "High"
    elif spo2 < 90:
        alerts.append("Warning: Low SpO2")
        if level != "Critical": level = "High"
    if emotion in ["anger", "fear", "sadness"] and level == "Normal":
        alerts.append(f"Negative Emotion: {emotion}"); level = "Moderate"
    if temp < 29 or temp > 37:
        alerts.append("Abnormal Temperature")
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
manual_fall_trigger: bool = False
frame_lock = threading.Lock()

def get_drawing_frame():
    global current_frame
    with frame_lock:
        if current_frame is None:
            return None
        return current_frame.copy()

async def generate_mjpeg():
    while True:
        frame = get_drawing_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.1) # ~10 FPS is plenty for MJPEG monitoring


# ── Camera thread ─────────────────────────────────────────────────────────────
def webcam_loop(emotion_model, emotion_mode, fall_model, fall_mode, fall_labels) -> None:
    global current_frame, manual_fall_trigger
    face_cascade, hog = load_detectors()
    pid = PATIENT_CONFIGS[0]["id"]
    
    cap = None
    for i in range(5):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            time.sleep(0.5) # Allow webcam to warm up
            cap = temp_cap
            logging.info(f"Camera found successfully at index {i}")
            break
        else:
            temp_cap.release()

    if cap is None:
        logging.warning("No camera found — vision AI disabled.")
        return

    fall_counter = 0
    last_capture_time = 0
    CAPTURE_COOLDOWN = 10 # seconds
    
    # Motion Detection variables
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    motion_history = []
    MOTION_FALL_THRESHOLD = 30000  # Increased significantly to ignore random noise/shadows
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. AI Detection
        fall_status, fall_conf = predict_fall(frame, hog, fall_model, fall_mode, fall_labels)
        # REQUIRE strict confidence (>= 75%) to prevent background from triggering false positive "falls"
        is_falling_ai = "fall" in str(fall_status).lower() and fall_conf >= 75.0
        
        # 2. Motion Detection (Heuristic for any falling object)
        fgmask = fgbg.apply(frame)
        # We look for large motion in the bottom half of the screen
        height, width = fgmask.shape
        bottom_half = fgmask[height//2:, :]
        motion_count = cv2.countNonZero(bottom_half)
        
        # Log motion for tuning
        if motion_count > 1000:
            logging.debug(f"[MOTION] Count: {motion_count}")

        # Heuristic: If sudden large motion in bottom half
        is_falling_motion = (motion_count > MOTION_FALL_THRESHOLD)
        
        # 3. Hybrid Logic: Either AI or Motion triggers the "Fallen" state
        is_falling = is_falling_ai or is_falling_motion or manual_fall_trigger
        
        if manual_fall_trigger:
            fall_counter = 20
            manual_fall_trigger = False
            logging.info("MANUAL FALL TRIGGERED VIA API")

        if is_falling:
            fall_counter = min(fall_counter + 1, 20) # Ramp up by +1 per frame so it requires consistent falling!
            if is_falling_motion:
                logging.info(f"[MOTION] Significant object/motion detected ({motion_count}px)")
        else:
            fall_counter = max(fall_counter - 1, 0)   # Decrease slower so temporary missed frame doesn't ruin it

        # 4. Draw Debug Info on Frame (Visible on Dashboard)
        debug_frame = frame.copy()
        color = (0, 0, 255) if is_falling else (0, 255, 0)
        cv2.putText(debug_frame, f"AI: {fall_status} ({fall_conf:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(debug_frame, f"Motion: {motion_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if fall_counter > 0:
            cv2.rectangle(debug_frame, (0, 0), (int(width * (fall_counter/20)), 10), (0, 0, 255), -1)

        # Update global frame for MJPEG stream
        with frame_lock:
            current_frame = debug_frame

        state = shared_state.get(pid, {})
        # Trigger alert if the counter reaches 10 (~1 full second of reliable fall detection)
        state["fall_alert"] = (fall_counter >= 10)
        if state["fall_alert"]:
            # Only log "ALERT TRIGGERED" when it's first activated
            if not state.get("was_logged"):
                logging.info("!!! FALL ALERT TRIGGERED !!!")
                state["was_logged"] = True
            fall_status = "Continuous Fall"
            
            # Capture logic
            now = time.time()
            if now - last_capture_time > CAPTURE_COOLDOWN:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(CAPTURES_DIR, f"fall_{ts}.jpg")
                cv2.imwrite(filename, frame)
                logging.info(f"FALL CAPTURED: {filename}")
                last_capture_time = now
        else:
            state["was_logged"] = False

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

                        # Store vitals in shared_state for broadcaster
                        if pid in shared_state:
                            shared_state[pid]["vitals"] = {"hr": hr, "spo2": spo2, "temp": temp, "crit": crit}

                        stage         = predict_sleep_stage(sleep_model, sleep_scaler, spo2, hr, temp)
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

                        # BROADCAST REMOVED FROM HERE
                        # await manager.broadcast(json.dumps(payload))
                        logging.debug(f"[MQTT] Update for {pid}")

                    except Exception as e:
                        logging.warning(f"[MQTT] Processing error: {e}")

        except aiomqtt.MqttError as e:
            logging.error(f"[MQTT] Broker error: {e}. Retrying in {reconnect_delay}s ...")
        await asyncio.sleep(reconnect_delay)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
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

    logging.info("Starting Periodic broadcaster ...")
    asyncio.create_task(periodic_broadcaster(sleep_model, sleep_scaler))
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/trigger_fall")
async def trigger_fall_api():
    global manual_fall_trigger
    manual_fall_trigger = True
    return {"status": "success"}


@app.get("/video_feed")
async def video_feed():
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


async def periodic_broadcaster(sleep_model, sleep_scaler):
    """Periodically broadcasts the full patient state (vitals + AI) to all clients."""
    logging.info("Broadcaster task started.")
    while True:
        try:
            for pid, patient_state in shared_state.items():
                # We need the latest vitals too. For now, since this script
                # doesn't persist raw vitals in shared_state, we only broadcast
                # when shared_state changes OR we can just broadcast whatever we have.
                # Actually, Server_Web proxies this, so we should broadcast AI + best-known vitals.
                
                # In the current architecture, Server_Pi_AI gets MQTT data.
                # Let's update shared_state with MQTT data so the broadcaster can use it.
                
                fall_status, fall_conf = patient_state.get("fall", ("No person", 0.0))
                emotion, emo_conf      = patient_state.get("emotion", ("neutral", 0.0))
                
                # Extract last vitals if we had them (this requires minor change in mqtt_listener)
                vitals = patient_state.get("vitals", {"hr": 0, "spo2": 0, "temp": 0.0, "crit": 0})
                hr, spo2, temp, crit = vitals["hr"], vitals["spo2"], vitals["temp"], vitals["crit"]
                
                sensors_connected = not (hr == 0 and spo2 == 0 and temp == 0)
                stage         = predict_sleep_stage(sleep_model, sleep_scaler, spo2, hr, temp)
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
                    "sensors_connected":  sensors_connected,
                    "timestamp":          datetime.datetime.now().strftime("%H:%M:%S"),
                }
                await manager.broadcast(json.dumps(payload))
        except Exception as e:
            logging.error(f"[BROADCASTER] Error: {e}")
        
        await asyncio.sleep(1) # Broadcast every 1 second


if __name__ == "__main__":
    print("=== Patient Monitor — Raspberry Pi AI Server ===")
    print(f"MQTT  : {MQTT_BROKER_HOST}:{MQTT_PORT}  topic: {MQTT_TOPIC_RAW}")
    print("Ready : ws://0.0.0.0:8001/ws")
    uvicorn.run(app, host="0.0.0.0", port=8001)
