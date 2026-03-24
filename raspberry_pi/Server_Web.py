import asyncio
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
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
import uvicorn
import websockets
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ── Configuration ─────────────────────────────────────────────────────────────
# Pi AI server runs on the same machine (localhost) — port 8001
PI_WEBSOCKET_URI = "ws://localhost:8001/ws"
LOCAL_PORT       = 8002

MONGODB_URL      = "mongodb://localhost:27017/"
DATABASE_NAME    = "patient_monitor_db"
COLLECTION_NAME  = "patient_records"

DUMMY_USER     = "admin"
DUMMY_PASSWORD = "password"

db_client: motor.motor_asyncio.AsyncIOMotorClient = None
db: motor.motor_asyncio.AsyncIOMotorDatabase       = None
last_pi_message_time: Optional[datetime]           = None


class PatientListItem(BaseModel):
    id: str = Field(..., alias="_id")
    patient_name: str


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


async def get_current_user(request: Request):
    token = request.cookies.get("session_token")
    if token != "fake-session-token":
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )
    return token


# ── Pi WebSocket proxy ────────────────────────────────────────────────────────
async def data_proxy_and_broadcast() -> None:
    global last_pi_message_time
    reconnect_delay = 5
    while True:
        try:
            logging.info(f"[PROXY] Connecting to Pi AI server at {PI_WEBSOCKET_URI} ...")
            async with websockets.connect(PI_WEBSOCKET_URI, open_timeout=10) as pi_ws:
                logging.info("[PROXY] Connected to Pi AI server. Relaying data ...")
                async for message in pi_ws:
                    last_pi_message_time = datetime.now()
                    await manager.broadcast(message)

                    if db is not None:
                        try:
                            data         = json.loads(message)
                            patient_id   = data.get("patient_id")
                            patient_name = data.get("patient_name", "Unknown")
                            if not patient_id:
                                continue
                            data["_server_timestamp"] = datetime.utcnow()
                            await db[COLLECTION_NAME].update_one(
                                {"_id": patient_id},
                                {"$push": {"data_records": data}, "$set": {"patient_name": patient_name}},
                                upsert=True,
                            )
                        except (json.JSONDecodeError, pymongo.errors.PyMongoError) as e:
                            logging.warning(f"[DB] Error: {e}")
        except Exception as e:
            logging.error(f"[PROXY] Error: {e}. Retrying in {reconnect_delay}s ...")
            last_pi_message_time = None
        await asyncio.sleep(reconnect_delay)


async def broadcast_network_health() -> None:
    global last_pi_message_time
    while True:
        await asyncio.sleep(5)
        mongo_latency_ms: Any = "N/A"
        pi_last_seen_s: Any   = "N/A"

        if db is not None:
            try:
                t0 = time.monotonic()
                await db.command("ping")
                mongo_latency_ms = (time.monotonic() - t0) * 1000
            except Exception:
                mongo_latency_ms = "FAIL"

        if last_pi_message_time:
            pi_last_seen_s = (datetime.now() - last_pi_message_time).total_seconds()

        try:
            await manager.broadcast(json.dumps({
                "type":           "system_health",
                "mongo_ping_ms":  mongo_latency_ms,
                "pi_last_seen_s": pi_last_seen_s,
            }))
        except Exception as e:
            logging.warning(f"[HEALTH] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, db
    logging.info("Web Dashboard starting up ...")
    try:
        db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        db = db_client[DATABASE_NAME]
        await db.command("ping")
        logging.info(f"MongoDB connected: '{DATABASE_NAME}'")
    except Exception as e:
        logging.error(f"MongoDB failed: {e}")
        db_client = db = None

    asyncio.create_task(data_proxy_and_broadcast())
    asyncio.create_task(broadcast_network_health())
    yield
    if db_client:
        db_client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/login")
async def get_login():
    return FileResponse("HTML/login.html")


@app.post("/login")
async def handle_login(username: str = Form(...), password: str = Form(...)):
    if username == DUMMY_USER and password == DUMMY_PASSWORD:
        resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        resp.set_cookie(key="session_token", value="fake-session-token", httponly=True)
        return resp
    return RedirectResponse(url="/login?error=1", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie(key="session_token")
    return resp


@app.get("/")
async def homepage(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/index.html")


@app.get("/records")
async def records_page(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/history.html")


@app.get("/api/patients", response_model=List[PatientListItem])
async def get_patients(user_token: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    patients = []
    async for p in db[COLLECTION_NAME].find({}, {"_id": 1, "patient_name": 1}):
        p["_id"] = str(p["_id"])
        patients.append(p)
    return patients


@app.get("/api/history_all")
async def get_history(user_token: str = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    all_data = []
    async for patient in db[COLLECTION_NAME].find({}):
        pid  = str(patient["_id"])
        name = patient.get("patient_name", "Unknown")
        for r in patient.get("data_records", []):
            all_data.append({
                "patient_id":        pid,
                "patient_name":      name,
                "heart_rate":        r.get("hr"),
                "oxygen":            r.get("spo2"),
                "temperature":       r.get("temp"),
                "_server_timestamp": (
                    r["_server_timestamp"].isoformat()
                    if isinstance(r.get("_server_timestamp"), datetime)
                    else r.get("_server_timestamp")
                ),
            })
    return {"records": all_data}


@app.get("/api/video")
async def video_proxy():
    """Proxies MJPEG stream from AI Server (8001) to Dashboard."""
    PI_VIDEO_URL = "http://localhost:8001/video_feed"
    
    async def stream_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("GET", PI_VIDEO_URL) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            except Exception as e:
                logging.error(f"[VIDEO PROXY] Failed to connect to {PI_VIDEO_URL}: {e}")

    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    if websocket.cookies.get("session_token") != "fake-session-token":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    print("=== Patient Monitor — Web Dashboard Server ===")
    print(f"Pi AI Source : {PI_WEBSOCKET_URI}")
    print(f"Dashboard    : http://0.0.0.0:{LOCAL_PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=LOCAL_PORT)
