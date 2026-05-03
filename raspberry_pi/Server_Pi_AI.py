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
import time
from typing import Dict, Any, Tuple, List, Optional
from sklearn.base import InconsistentVersionWarning
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import uvicorn
import aiomqtt
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import sqlite3
import subprocess
import aiofiles

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not available. Hardware buzzer will be mocked (ideal for Windows/testing).")

# Hardware Configuration
BUZZER_PIN = 18

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# ── Store and Forward DB Setup ────────────────────────────────────────────────
DB_NAME = "offline_buffer.db"
def init_db():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS offline_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    payload TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    except Exception as e:
        logging.error(f"Failed to initialize offline DB: {e}")

# --- Local Keyboard Trigger for Viva ---
import termios
import tty
def handle_local_keypress():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
    if char.lower() == 'f':
        logging.info("[VIVA] LOCAL FALL TRIGGER DETECTED (Pi Keyboard)!")
        for pid in shared_state:
            shared_state[pid]["manual_fall_active"] = True
            shared_state[pid]["fall_alert"] = True
            shared_state[pid]["fall"] = ("FALL DETECTED", 100.0)
            if GPIO_AVAILABLE:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                # Auto-reset after 5 seconds
                threading.Timer(5.0, lambda: GPIO.output(BUZZER_PIN, GPIO.LOW)).start()

init_db()

NETWORK_STATUS = "EXCELLENT"
ACTIVE_NETWORK = "Primary WiFi"

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
MQTT_BROKER_HOST = "broker.hivemq.com"
MQTT_PORT        = 1883
MQTT_TOPIC_RAW   = "r25_014/vitals/raw"
MQTT_TOPIC_PROCESSED = "r25_014/vitals/processed"
MQTT_TOPIC_STATUS = "r25_014/vitals/status"

PATIENT_CONFIGS = [
    {"id": "P001", "name": "Patient Fernando"}
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
        if os.path.exists(h5_path):
            try:
                model = tf.keras.models.load_model(h5_path, compile=False)
                logging.info(f"Keras model loaded: {h5_path}")
                return model, "keras"
            except Exception as e:
                logging.warning(f"Keras load failed for {h5_path}: {e}")

    if TFLITE_AVAILABLE and TFLiteInterpreter is not None:
        if os.path.exists(tflite_path):
            try:
                interp = TFLiteInterpreter(model_path=tflite_path)
                interp.allocate_tensors()
                logging.info(f"TFLite model loaded: {tflite_path}")
                return interp, "tflite"
            except Exception as e:
                logging.warning(f"TFLite load failed for {tflite_path}: {e}")
    
    logging.error(f"FATAL: All model loading attempts failed for {h5_path}")
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


def predict_fall(frame: np.ndarray, hog, model, mode: str, labels, fallback_detected: bool = False) -> Tuple[str, float]:
    try:
        # Evaluate HOG with more sensitive scale to find people in unusual (fallen) positions
        (boxes, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.03)
        best_box = None
        best_weight = 0.0
        
        if len(boxes) > 0 and len(weights) > 0:
            for (x, y, w, h), weight in zip(boxes, weights):
                w_val = weight[0] if isinstance(weight, (list, np.ndarray)) else weight
                if w_val > 0.4 and w_val > best_weight:
                    best_weight = w_val
                    best_box = (x, y, w, h)
                    
        if model is None or labels is None:
            return "Unknown", 0.0
            
        # Neural pipeline requires RGB format, whereas OpenCV uses BGR natively.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ISOLATE HUMAN FIGURE (Crop using HOG bounding box) to eliminate background noise
        if best_box is not None:
            (x, y, w, h) = best_box
            # Add a small margin to the bounding box
            margin = int(0.1 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            person_crop = rgb_frame[y1:y2, x1:x2]
        else:
            person_crop = rgb_frame # Fallback if only face was detected but no HOG body
            
        resized = cv2.resize(person_crop, (224, 224)).astype(np.float32)
        arr     = ((resized / 127.5) - 1.0).reshape(1, 224, 224, 3)
        pred    = run_vision_inference(model, mode, arr)
        
        if pred.size == 0:
            return "No person", 0.0
            
        idx = int(np.argmax(pred[0]))
        raw_label = labels[idx] if idx < len(labels) else "Unknown"
        conf = float(pred[0][idx] * 100)
        
        idx = int(np.argmax(pred[0]))
        raw_label = labels[idx] if idx < len(labels) else "Unknown"
        conf = float(pred[0][idx] * 100)
        
        raw_upper = raw_label.upper()
        if "NOT" in raw_upper or "NO" in raw_upper:
            final_label = "Normal"
        elif "FALL" in raw_upper:
            final_label = "A FALL HAS OCCURED"
        else:
            final_label = raw_label
            
        return final_label, conf
    except Exception as e:
        logging.warning(f"Fall inference error: {e}")
        return "No person", 0.0


def predict_sleep_stage(model, scaler, spo2: int, hr: int, temp: float) -> str:
    try:
        # 1. Hardware/Sensor Disconnect Guard
        if hr == 0 or spo2 == 0:
            return "Awake"
            
        # 2. Physiological Absolute Overrides (Overrides AI Model)
        # If heart rate is very high, it is physiologically impossible to be in Deep sleep.
        if hr > 85:
            return "Awake"
            
        if model is None or scaler is None:
            return "Awake"
            
        # 3. AI Model Prediction
        features = pd.DataFrame([[spo2, hr, temp]], columns=["spo2", "hr", "temp"])
        prediction = str(model.predict(scaler.transform(features))[0])
        
        # 4. AI Guardrails & Remapping (Correcting Model Bias)
        if prediction == "Deep" and hr > 75:
            return "Light" # Corrects biased models that predict Deep despite elevated HR
            
        # The AI model sometimes outputs "Warm-up" or "Warmup" for uncertain data.
        # We remap this to "Awake" so the dashboard doesn't get stuck showing "Analyzing..."
        if "warm" in prediction.lower():
            return "Awake"
            
        return prediction
    except Exception as e:
        return "Awake"


def evaluate_criticality(hr, spo2, temp, stage, fall_status, emotion, esp_crit) -> Tuple[str, str]:
    alerts, level = [], "Normal"
    if esp_crit == 1:
        alerts.append("SENSOR CRITICAL THRESHOLD"); level = "Critical"
    if isinstance(fall_status, str) and "fall" in fall_status.lower():
        alerts.append("AI FALL DETECTED"); level = "Critical"
        
    if hr > 0:
        if hr > 100:
            alerts.append(f"Abnormal Heart Rate: Too High ({hr} BPM)")
            if level != "Critical": level = "High"
        elif hr < 55:
            alerts.append(f"Abnormal Heart Rate: Too Low ({hr} BPM)")
            if level != "Critical": level = "High"
            
    if spo2 > 0:
        if spo2 < 90:
            alerts.append(f"Warning: Low SpO2 ({spo2}%)")
            if level != "Critical": level = "High"
    if emotion in ["anger", "fear", "sadness"] and level == "Normal":
        alerts.append(f"Negative Emotion: {emotion}"); level = "Moderate"
    if temp > 38.5:
        alerts.append(f"CRITICAL Temperature ({temp}°C)")
        level = "Critical"
    elif temp > 37.5:
        alerts.append(f"Elevated Temperature ({temp}°C)")
        if level != "Critical": level = "High"
    elif temp < 35.5:
        alerts.append(f"Low Temperature ({temp}°C)")
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

shared_state = { cfg["id"]: {"fall": ("No person", 0.0), "emotion": ("neutral", 0.0), "fall_alert": False, "manual_fall_active": False} for cfg in PATIENT_CONFIGS }
manual_override_until = 0

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
            # Dynamic compression based on NETWORK_STATUS
            global NETWORK_STATUS
            quality = 70
            sleep_time = 0.1 # 10 FPS
            if NETWORK_STATUS == "FAIR":
                quality = 50
                sleep_time = 0.2 # 5 FPS
            elif NETWORK_STATUS == "POOR":
                quality = 30
                sleep_time = 0.5 # 2 FPS

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(sleep_time)


# ── Camera thread ─────────────────────────────────────────────────────────────
def webcam_loop(emotion_model, emotion_mode, fall_model, fall_mode, fall_labels) -> None:
    global CAMERA_ACTIVE
    face_cascade, hog = load_detectors()
    pid = PATIENT_CONFIGS[0]["id"]
    # --- Improved Camera Discovery ---
    cap = None
    for i in range(5):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            time.sleep(0.5) # Warm up
            cap = temp_cap
            logging.info(f"Camera found successfully at index {i}")
            break
        else:
            temp_cap.release()

    if cap is None:
        logging.warning("No camera found — vision AI disabled.")
        CAMERA_ACTIVE = False
        return
    
    CAMERA_ACTIVE = True

    # --- Motion Detection setup ---
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    MOTION_FALL_THRESHOLD = 5000  # Lowered from 20000 for better sensitivity

    fall_counter = 0
    miss_counter = 0
    MISS_THRESHOLD = 5  # Reduced from 20 for much snappier 'No Person' transitions
    fall_start_time = None
    manual_override_until = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            logging.warning("Webcam frame lost. Attempting to reconnect...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(0) # Re-init
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Verify face pipeline for strong presence weighting
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        face_detected = len(faces) > 0
        
        # --- Fall Detection (AI + Motion) ---
        fall_status, fall_conf = predict_fall(frame, hog, fall_model, fall_mode, fall_labels, fallback_detected=face_detected)
        
        # Motion Check (as backup/heuristic)
        fgmask = fgbg.apply(frame)
        height, width = fgmask.shape
        bottom_half = fgmask[height//2:, :]
        motion_count = cv2.countNonZero(bottom_half)
        
        is_falling_ai = "fall" in str(fall_status).lower() and fall_conf >= 35.0
        is_falling_motion = motion_count > MOTION_FALL_THRESHOLD
        
        # Smart Consolidation
        if is_falling_ai and is_falling_motion:
            fall_status = "FALL DETECTED (High Conf)"
            is_falling = True
        elif is_falling_ai:
            fall_status = "POTENTIAL FALL (Low Conf)"
            is_falling = True
        elif is_falling_motion:
            fall_status = "Motion Fall detected"
            is_falling = True
        else:
            if "No person" in fall_status:
                fall_status = "Normal"
            is_falling = False

        state = shared_state.get(pid, {})
        
        # --- NEW: Manual Override Logic ---
        if time.time() < manual_override_until:
            is_falling = True
            fall_status = "A FALL HAS OCCURED (Manual)"
        elif state.get("manual_fall_active"):
            # Triggered from MQTT listener
            manual_override_until = time.time() + 8.0
            state["manual_fall_active"] = False # Reset flag
            is_falling = True
            fall_status = "A FALL HAS OCCURED (Manual)"

        if is_falling:
            if fall_start_time is None:
                fall_start_time = time.time()
            
            elapsed = time.time() - fall_start_time
            if elapsed >= 2.0: # Reduced to 2 seconds for better responsiveness
                if "fall" in str(fall_status).lower():
                    state["fall_alert"] = True
                    # TRIGGER HARDWARE BUZZER (GPIO)
                    fall_status = "FALL DETECTED"
                else:
                    state["fall_alert"] = False
        else:
            fall_start_time = None
            state["fall_alert"] = False

        # 4. Handle Hardware Buzzer based on Fall Status
        if GPIO_AVAILABLE:
            # Check if a fall is detected by AI OR if a Manual Fall is active
            is_ai_fall = "fall" in str(fall_status).lower()
            is_manual_fall = any(state.get("manual_fall_active", False) for state in shared_state.values())
            
            if is_ai_fall or is_manual_fall or (time.time() < manual_override_until):
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
            else:
                GPIO.output(BUZZER_PIN, GPIO.LOW)

        # 5. Emotion Detection
        if face_detected:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            emotion, emo_conf = predict_emotion(frame[y:y+h, x:x+w], emotion_model, emotion_mode)
        else:
            emotion, emo_conf = "No face", 0.0

        # 6. Update Shared State
        state["fall"]    = (fall_status, fall_conf)
        state["emotion"] = (emotion, emo_conf)
        shared_state[pid] = state

        # Update global frame for streaming
        global current_frame
        # --- Draw Labels for Video Feed ---
        debug_frame = frame.copy()
        color = (0, 0, 255) if ("fall" in str(fall_status).lower()) else (0, 255, 0)
        cv2.putText(debug_frame, f"Fall: {fall_status} ({fall_conf:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(debug_frame, f"Emo: {emotion} ({emo_conf:.1f}%)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        with frame_lock:
            current_frame = debug_frame


# ── MQTT listener ─────────────────────────────────────────────────────────────
async def mqtt_listener(sleep_model, sleep_scaler) -> None:
    reconnect_delay = 5
    while True:
        try:
            logging.info(f"[MQTT] Connecting to {MQTT_BROKER_HOST}:{MQTT_PORT} ...")
            async with aiomqtt.Client(hostname=MQTT_BROKER_HOST, port=MQTT_PORT) as client:
                await client.subscribe(MQTT_TOPIC_RAW)
                await client.subscribe(MQTT_TOPIC_STATUS)
                logging.info(f"[MQTT] Connected. Subscribed to '{MQTT_TOPIC_RAW}' and '{MQTT_TOPIC_STATUS}'")
                async for message in client.messages:
                    try:
                        logging.info(f"[MQTT RECV] Topic: {message.topic}")
                        raw  = json.loads(message.payload)
                        pid  = str(raw.get("pid", PATIENT_CONFIGS[0]["id"]))
                        hr   = int(raw.get("hr",   0))
                        spo2 = int(raw.get("spo2", 0))
                        temp = float(raw.get("temp", 0.0))
                        crit = int(raw.get("crit",  0))
                        manual_trigger = raw.get("manual_fall", False)

                        # Hybrid Logic: Use provided stage (e.g. from Mock/Sensor) if available, otherwise predict via AI
                        stage = raw.get("stage")
                        if not stage or stage == "Warmup":
                            stage = predict_sleep_stage(sleep_model, sleep_scaler, spo2, hr, temp)
                        patient_state = shared_state.get(pid, shared_state[PATIENT_CONFIGS[0]["id"]])
                        
                        # Handle Manual Viva Trigger
                        if manual_trigger:
                            global manual_override_until
                            logging.info("[VIVA] MANUAL FALL TRIGGER ACTIVATED via Keyboard!")
                            patient_state["manual_fall_active"] = True 
                            patient_state["fall_alert"] = True
                            patient_state["fall"] = ("FALL DETECTED", 100.0)
                            manual_override_until = time.time() + 5.0 # Lock buzzer ON for 5s
                            if GPIO_AVAILABLE:
                                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                                # Failsafe: Ensure state resets after 5 seconds
                                threading.Timer(5.0, lambda: patient_state.update({"manual_fall_active": False, "fall_alert": False})).start()

                        fall_status, fall_conf = patient_state.get("fall", ("No person", 0.0))
                        emotion, emo_conf      = patient_state.get("emotion", ("neutral", 0.0))
                        if patient_state.get("fall_alert"):
                            fall_status = "FALL DETECTED"

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
                            "fall_status":        "FALL DETECTED" if "fall" in str(fall_status).lower() else fall_status,
                            "emotion_confidence": emo_conf,
                            "fall_confidence":    fall_conf,
                            "timestamp":          datetime.datetime.now().strftime("%H:%M:%S"),
                        }

                        payload_str = json.dumps(payload)
                        await manager.broadcast(payload_str)
                        try:
                            await client.publish(MQTT_TOPIC_PROCESSED, payload=payload_str)
                            logging.info(f"[BROADCAST & PUBLISH] {pid} HR:{hr} SpO2:{spo2} Temp:{temp} → {level}")
                        except Exception as pub_e:
                            logging.warning(f"[MQTT] Publish failed, buffering locally: {pub_e}")
                            with sqlite3.connect(DB_NAME) as conn:
                                conn.cursor().execute("INSERT INTO offline_messages (topic, payload) VALUES (?, ?)", (MQTT_TOPIC_PROCESSED, payload_str))
                                conn.commit()

                    except Exception as e:
                        logging.warning(f"[MQTT] Processing error: {e}")

        except aiomqtt.MqttError as e:
            logging.error(f"[MQTT] Broker connection lost: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
        except Exception as e:
            logging.error(f"[MQTT] Unexpected error in listener: {e}. Restarting...")
            await asyncio.sleep(reconnect_delay)

# ── Network Monitoring & Persistence Tasks ────────────────────────────────────
async def sync_offline_data():
    while True:
        await asyncio.sleep(10)
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, topic, payload FROM offline_messages ORDER BY id ASC LIMIT 50")
                rows = cursor.fetchall()
            
            if rows:
                async with aiomqtt.Client(hostname=MQTT_BROKER_HOST, port=MQTT_PORT) as client:
                    for row_id, topic, payload in rows:
                        await client.publish(topic, payload=payload)
                        with sqlite3.connect(DB_NAME) as conn:
                            conn.cursor().execute("DELETE FROM offline_messages WHERE id = ?", (row_id,))
                            conn.commit()
                    logging.info(f"[SYNC] Successfully synced {len(rows)} offline messages.")
        except Exception:
            pass # Ignore if broker is still down

async def network_health_monitor():
    global NETWORK_STATUS, ACTIVE_NETWORK
    while True:
        await asyncio.sleep(1)
        # Check network fallback state
        try:
            fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fallback_state.json")
            if os.path.exists(fallback_path):
                with open(fallback_path, "r") as f:
                    state = json.load(f)
                    ACTIVE_NETWORK = state.get("active_network", "Primary WiFi")
        except Exception:
            pass

        # Ping main server to check latency
        try:
            # We use a simple ping to localhost since MQTT broker is local in this setup, 
            # or ping 8.8.8.8 to check internet latency.
            if sys.platform == 'win32':
                cmd = ["ping", "-n", "1", "8.8.8.8"]
            else:
                cmd = ["ping", "-c", "1", "-W", "1", "8.8.8.8"]
                
            start = time.time()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await proc.communicate()
            latency = (time.time() - start) * 1000

            if proc.returncode != 0:
                NETWORK_STATUS = "POOR"
            elif latency > 500:
                NETWORK_STATUS = "POOR"
            elif latency > 200:
                NETWORK_STATUS = "FAIR"
            else:
                NETWORK_STATUS = "EXCELLENT"
        except Exception as e:
            NETWORK_STATUS = "POOR"

async def broadcast_telemetry():
    global NETWORK_STATUS, ACTIVE_NETWORK
    import aiomqtt
    will = aiomqtt.Will(topic=MQTT_TOPIC_STATUS, payload=b'{"device": "pi", "status": "offline"}', qos=1, retain=True)
    
    last_published_network = ACTIVE_NETWORK
    last_publish_time = 0
    
    while True:
        try:
            async with aiomqtt.Client(hostname=MQTT_BROKER_HOST, port=MQTT_PORT, will=will) as client:
                await client.publish(MQTT_TOPIC_STATUS, payload=b'{"device": "pi", "status": "online"}', qos=1, retain=True)
                logging.info(f"[LWT] Persistent MQTT connection established with Will on {MQTT_TOPIC_STATUS}")
                
                while True:
                    await asyncio.sleep(1)
                    current_time = time.time()
                    network_changed = (ACTIVE_NETWORK != last_published_network)
                    
                    if current_time - last_publish_time >= 5 or network_changed:
                        try:
                            buffer_size = 0
                            with sqlite3.connect(DB_NAME) as conn:
                                cursor = conn.cursor()
                                cursor.execute("SELECT COUNT(*) FROM offline_messages")
                                buffer_size = cursor.fetchone()[0]

                            payload = {
                                "type": "network_telemetry",
                                "buffer_size": buffer_size,
                                "active_network": ACTIVE_NETWORK,
                                "network_status": NETWORK_STATUS
                            }
                            payload_str = json.dumps(payload)
                            await manager.broadcast(payload_str)
                            await client.publish("r25_014/vitals/telemetry", payload=payload_str)
                            
                            last_publish_time = current_time
                            last_published_network = ACTIVE_NETWORK
                            
                        except Exception as e:
                            logging.error(f"Telemetry error: {e}")
        except Exception as e:
            logging.error(f"MQTT Client connection failed, retrying in 5s... {e}")
            await asyncio.sleep(5)


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
    loop = asyncio.get_event_loop()
    
    # Only listen to local keyboard if we are in a real terminal (TTY)
    if sys.stdin.isatty():
        try:
            loop.add_reader(sys.stdin, handle_local_keypress)
            logging.info("Local keyboard listener active (Press 'F' for manual trigger).")
        except Exception as e:
            logging.warning(f"Could not start local keyboard listener: {e}")
    else:
        logging.info("Background mode: Local keyboard listener disabled.")
    
    asyncio.create_task(mqtt_listener(sleep_model, sleep_scaler))
    asyncio.create_task(sync_offline_data())
    asyncio.create_task(network_health_monitor())
    asyncio.create_task(broadcast_telemetry())
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)


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


if __name__ == "__main__":
    if sys.platform == 'win32':
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()

    config = uvicorn.Config(app, host="0.0.0.0", port=8001, loop="none")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
