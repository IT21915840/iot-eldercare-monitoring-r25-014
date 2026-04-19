import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import logging
from typing import List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
import time
import motor.motor_asyncio
import pymongo.errors
from pydantic import BaseModel, Field
import pytz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, Form, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
import aiomqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- MQTT / ESP32 Configuration ---
MQTT_BROKER_IP  = "localhost"   # Changed to localhost for local testing
MQTT_PORT       = 1883
MQTT_TOPIC_PROCESSED = "vitals/processed"

# --- Web Server ---
LOCAL_PORT = 8002

# --- MongoDB ---
MONGODB_URL     = "mongodb://localhost:27017/"
DATABASE_NAME   = "patient_monitor_db"
COLLECTION_NAME = "patient_records"

# --- Auth ---
DUMMY_USER     = "admin"
DUMMY_PASSWORD = "password"

db_client: motor.motor_asyncio.AsyncIOMotorClient = None
db: motor.motor_asyncio.AsyncIOMotorDatabase = None

last_mqtt_message_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PatientListItem(BaseModel):
    id: str = Field(..., alias="_id")
    patient_name: str


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def broadcast(self, message: str):
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                to_remove.append(connection)
        for connection in to_remove:
            self.disconnect(connection)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# MQTT listener task  (replaces old Pi WebSocket proxy)
# ---------------------------------------------------------------------------
async def mqtt_listener() -> None:
    global last_mqtt_message_time
    reconnect_delay = 5
    while True:
        try:
            logging.info(f"[MQTT] Connecting to broker at {MQTT_BROKER_IP}:{MQTT_PORT} ...")
            async with aiomqtt.Client(hostname=MQTT_BROKER_IP, port=MQTT_PORT) as client:
                logging.info(f"[MQTT] Connected. Subscribing to '{MQTT_TOPIC_PROCESSED}' ...")
                await client.subscribe(MQTT_TOPIC_PROCESSED)
                async for message in client.messages:
                    last_mqtt_message_time = datetime.now()
                    try:
                        raw = json.loads(message.payload)
                        # We now expect a full payload from Server_Pi_AI.py instead of raw ESP32 data.
                        dashboard_data = raw
                        payload_str = json.dumps(dashboard_data)

                        # Push to all browser clients
                        await manager.broadcast(payload_str)

                        # Persist to MongoDB
                        if db is not None:
                            try:
                                patient_id   = dashboard_data["patient_id"]
                                patient_name = dashboard_data["patient_name"]
                                record = {**dashboard_data, "_server_timestamp": datetime.utcnow()}
                                collection = db[COLLECTION_NAME]
                                await collection.update_one(
                                    {"_id": patient_id},
                                    {
                                        "$push": {"data_records": record},
                                        "$set":  {"patient_name": patient_name},
                                    },
                                    upsert=True,
                                )
                            except pymongo.errors.PyMongoError as e:
                                logging.error(f"[DB] MongoDB error: {e}")
                        else:
                            logging.warning("[DB] No database connection, skipping save.")

                    except (json.JSONDecodeError, Exception) as e:
                        logging.warning(f"[MQTT] Failed to process message: {e}")

        except aiomqtt.MqttError as e:
            logging.error(f"[MQTT] Connection error: {e}. Retrying in {reconnect_delay}s ...")
            last_mqtt_message_time = None
        except Exception as e:
            logging.error(f"[MQTT] Unexpected error: {e}. Retrying in {reconnect_delay}s ...")
            last_mqtt_message_time = None

        await asyncio.sleep(reconnect_delay)


# ---------------------------------------------------------------------------
# System health broadcaster  (now reports MQTT last-seen instead of Pi)
# ---------------------------------------------------------------------------
async def broadcast_network_health() -> None:
    global last_mqtt_message_time
    while True:
        await asyncio.sleep(5)
        mongo_latency_ms: Any = "N/A"
        mqtt_last_seen_s: Any = "N/A"

        if db is not None:
            try:
                start_time = time.monotonic()
                await db.command("ping")
                mongo_latency_ms = (time.monotonic() - start_time) * 1000
            except Exception:
                mongo_latency_ms = "FAIL"

        if last_mqtt_message_time:
            delta = datetime.now() - last_mqtt_message_time
            mqtt_last_seen_s = delta.total_seconds()

        try:
            payload = {
                "type":           "system_health",
                "mongo_ping_ms":  mongo_latency_ms,
                "pi_last_seen_s": mqtt_last_seen_s,   # key kept same so dashboard JS needs no change
            }
            await manager.broadcast(json.dumps(payload))
        except Exception as e:
            logging.warning(f"[HEALTH] Error broadcasting system health: {e}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, db
    logging.info("Web Dashboard Host starting up...")
    try:
        logging.info(f"Connecting to MongoDB at {MONGODB_URL} ...")
        db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        db = db_client[DATABASE_NAME]
        await db.command("ping")
        logging.info(f"Successfully connected to MongoDB. Database: '{DATABASE_NAME}'")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        db_client = None
        db = None

    asyncio.create_task(mqtt_listener())
    asyncio.create_task(broadcast_network_health())
    yield

    logging.info("Web Dashboard Host shutting down...")
    if db_client:
        db_client.close()
        logging.info("MongoDB connection closed.")


app = FastAPI(lifespan=lifespan)
app.mount("/HTML", StaticFiles(directory="HTML"), name="HTML")

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
async def get_current_user(request: Request):
    token = request.cookies.get("session_token")
    if token != "fake-session-token":
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"},
        )
    return token


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/login")
async def get_login_page():
    return FileResponse("HTML/login.html")


@app.post("/login")
async def handle_login(response: Response, username: str = Form(...), password: str = Form(...)):
    if username == DUMMY_USER and password == DUMMY_PASSWORD:
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="session_token", value="fake-session-token", httponly=True)
        return response
    return RedirectResponse(url="/login?error=1", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="session_token")
    return response


@app.get("/")
async def get_homepage(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/index.html")


@app.get("/network")
async def get_network_page(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/network.html")


@app.get("/records")
async def get_records_page(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/history.html")


@app.get("/api/patients", response_model=List[PatientListItem])
async def get_all_patients(user_token: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    patient_list = []
    cursor = db[COLLECTION_NAME].find({}, {"_id": 1, "patient_name": 1})
    async for patient in cursor:
        patient["_id"] = str(patient["_id"])
        patient_list.append(patient)
    return patient_list


@app.get("/api/history_all")
async def get_all_patient_records(user_token: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    all_data = []
    cursor = db[COLLECTION_NAME].find({})
    async for patient in cursor:
        patient_id   = str(patient["_id"])
        patient_name = patient.get("patient_name", "Unknown")
        for record in patient.get("data_records", []):
            # Robustly capture the timestamp for sorting/display
            raw_ts = record.get("_server_timestamp")
            if isinstance(raw_ts, datetime):
                # Ensure UTC marker 'Z' for frontend local conversion
                iso_ts = raw_ts.isoformat() + "Z"
            elif isinstance(raw_ts, str):
                iso_ts = raw_ts if raw_ts.endswith("Z") else raw_ts + "Z"
            else:
                iso_ts = "0000-00-00T00:00:00Z" # Fallback for missing timestamps

            record_entry = {
                "patient_id":        patient_id,
                "patient_name":      patient_name,
                "heart_rate":        record.get("hr"),
                "oxygen":            record.get("spo2"),
                "temperature":       record.get("temp"),
                "emotion":           record.get("emotion"),
                "fall_status":       record.get("fall_status"),
                "stage":             record.get("stage"),
                "_server_timestamp": iso_ts,
            }
            all_data.append(record_entry)
            
    # Sort by timestamp descending (newest first)
    # Using a list comprehension to ensure we only sort valid ISO strings
    all_data.sort(key=lambda x: x["_server_timestamp"] if x["_server_timestamp"] else "", reverse=True)
    if all_data:
        logging.info(f"[API] History sorted. Top record: {all_data[0]['_server_timestamp']}")
    return {"records": all_data}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.cookies.get("session_token")
    if token != "fake-session-token":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logging.warning("WebSocket connection rejected: No valid session token.")
        return
    await manager.connect(websocket)
    logging.info("Browser dashboard client connected. Hydrating recent state...")
    
    # Safely hydrate the dashboard with the last 15 events to prevent empty layout
    try:
        if db is not None:
            patient = await db[COLLECTION_NAME].find_one({"_id": "P001"})
            if patient and "data_records" in patient:
                recent = patient["data_records"][-15:]
                for rec in recent:
                    payload = dict(rec)
                    # Exclude backend-only properties if necessary
                    if "_server_timestamp" in payload:
                        del payload["_server_timestamp"]
                    try:
                        await websocket.send_text(json.dumps(payload))
                        await asyncio.sleep(0.05) # Yield event loop & create sequential UI loading animation
                    except Exception:
                        pass
    except Exception as e:
        logging.warning(f"State hydration failed: {e}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logging.info("Browser dashboard client disconnected.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if sys.platform == 'win32':
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()

    config = uvicorn.Config(app, host="0.0.0.0", port=LOCAL_PORT, loop="none")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
