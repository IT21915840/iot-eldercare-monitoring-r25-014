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
import subprocess
import os
from pydantic import BaseModel, Field
import pytz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, Form, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
import aiomqtt
import httpx
from fastapi.responses import StreamingResponse
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import sqlite3
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- MQTT / ESP32 Configuration ---
MQTT_BROKER_IP  = "broker.hivemq.com"   # Cloud MQTT Broker for true cross-network failover
MQTT_PORT       = 1883
MQTT_TOPIC_PROCESSED = "r25_014/vitals/processed"

# --- Web Server ---
LOCAL_PORT = 8002
# Secure Access via Tailscale: 100.94.199.97

# --- SQLite Configuration (HIPAA Secure Edge Storage) ---
# Hardcoded for the Raspberry Pi demonstration to ensure no path mismatches
DB_FILE = "/home/user/Desktop/iot-eldercare-monitoring-r25-014/eldercare_secure.db"

def sync_init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            hr INTEGER,
            spo2 INTEGER,
            temp REAL,
            stage TEXT,
            fall_status TEXT,
            emotion TEXT,
            alerts TEXT,
            level TEXT,
            timestamp TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS allowed_ips (
            ip_address TEXT PRIMARY KEY,
            description TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT
        )
    """)
    conn.commit()
    conn.close()

async def init_db():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sync_init_db)

# --- Auth ---
DUMMY_USER     = "admin"
DUMMY_PASSWORD = "password"

# --- Process Manager ---
ACTIVE_PROCESSES = {
    "ai": None,
    "network": None
}

last_mqtt_message_time: Optional[datetime] = None
DB_OFFLINE_BUFFER: List[dict] = []
ACTIVE_SESSIONS = {}  # { token: {"username": str, "role": str} }


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
            async with aiomqtt.Client(MQTT_BROKER_IP, port=MQTT_PORT) as client:
                logging.info(f"[MQTT] Connected to Cloud Broker ({MQTT_BROKER_IP}). Subscribing to '{MQTT_TOPIC_PROCESSED}' and 'r25_014/vitals/telemetry' and 'r25_014/vitals/status' ...")
                await client.subscribe(MQTT_TOPIC_PROCESSED)
                await client.subscribe("r25_014/vitals/telemetry")
                await client.subscribe("r25_014/vitals/status")
                async for message in client.messages:
                    last_mqtt_message_time = datetime.now()
                    try:
                        raw = json.loads(message.payload)
                        
                        # Handle Telemetry Payload separately
                        if raw.get("type") == "network_telemetry":
                            await manager.broadcast(json.dumps(raw))
                            continue
                            
                        # Handle LWT Status explicitly
                        if message.topic.value == "r25_014/vitals/status":
                            if raw.get("status") == "offline" and raw.get("device") == "pi":
                                await manager.broadcast(json.dumps({"type": "node_offline", "device": "pi"}))
                            continue

                        # We now expect a full payload from Server_Pi_AI.py instead of raw ESP32 data.
                        dashboard_data = raw
                        payload_str = json.dumps(dashboard_data)

                        # Push to all browser clients
                        await manager.broadcast(payload_str)

                        # Push to all browser clients
                        await manager.broadcast(payload_str)

                        # Persist to SQLite (Secure Edge Storage)
                        def save_to_db(data):
                            conn = sqlite3.connect(DB_FILE)
                            c = conn.cursor()
                            
                            # Smoothing: Filter out noise (300 BPM jumps)
                            hr = data.get("hr", 0)
                            if hr > 180 or hr < 30:
                                conn.close()
                                return
                                
                            c.execute("""
                                INSERT INTO patient_records (
                                    patient_id, patient_name, hr, spo2, temp, 
                                    stage, fall_status, emotion, alerts, level, timestamp
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                data.get("patient_id") or data.get("pid", "P001"),
                                data.get("patient_name") or "Fernando",
                                data.get("hr", 0),
                                data.get("spo2", 0),
                                data.get("temp", 0.0),
                                data.get("stage", "Unknown"),
                                data.get("fall_status", "Normal"),
                                data.get("emotion", "neutral"),
                                data.get("alerts", ""),
                                data.get("level", "NORMAL"),
                                data.get("timestamp") or datetime.now().isoformat()
                            ))
                            conn.commit()
                            conn.close()

                        try:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, save_to_db, dashboard_data)
                        except Exception as e:
                            logging.error(f"[DB] SQLite error: {e}")

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
        db_status: Any = "OFFLINE"
        mqtt_last_seen_s: Any = "N/A"

        # Check SQLite Status
        def check_db_sync():
            if not os.path.exists(DB_FILE): return "INITIALIZING"
            try:
                conn = sqlite3.connect(DB_FILE)
                conn.execute("SELECT 1")
                conn.close()
                return "SYNCED"
            except: return "ERROR"

        loop = asyncio.get_event_loop()
        db_status = await loop.run_in_executor(None, check_db_sync)

        if last_mqtt_message_time:
            delta = datetime.now() - last_mqtt_message_time
            mqtt_last_seen_s = delta.total_seconds()

        try:
            # Check status of background processes
            services = {}
            for svc, proc in ACTIVE_PROCESSES.items():
                if proc is not None and proc.poll() is None:
                    services[svc] = "running"
                else:
                    services[svc] = "stopped"

            payload = {
                "type":           "system_health",
                "db_status":      db_status,
                "pi_last_seen_s": mqtt_last_seen_s,
                "services":       services
            }
            await manager.broadcast(json.dumps(payload))
        except Exception as e:
            logging.warning(f"[HEALTH] Error broadcasting system health: {e}")


async def broadcast_network_telemetry() -> None:
    state_file = os.path.join("raspberry_pi", "fallback_state.json")
    global last_mqtt_message_time
    
    while True:
        await asyncio.sleep(5)
        active_network = "--"
        network_status = "NORMAL"
        buffer_size = len(DB_OFFLINE_BUFFER)

        # Read active network from fallback daemon state file
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    active_network = state.get("active_network", "--")
            except Exception:
                pass

        # Determine network quality based on MQTT delay
        if last_mqtt_message_time:
            delta = (datetime.now() - last_mqtt_message_time).total_seconds()
            if delta > 30:
                network_status = "POOR"
            elif delta > 10:
                network_status = "FAIR"
            else:
                network_status = "EXCELLENT"
        else:
            network_status = "OFFLINE"

        try:
            payload = {
                "type":           "network_telemetry",
                "active_network": active_network,
                "network_status": network_status,
                "buffer_size":    buffer_size
            }
            await manager.broadcast(json.dumps(payload))
        except Exception as e:
            logging.warning(f"[TELEMETRY] Error broadcasting network telemetry: {e}")


# SQLite is local and persistent, so we don't need a background watchdog for cloud sync.


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Web Dashboard Host starting up...")
    
    # Initialize Ngrok
    public_url = None
    try:
        if NGROK_AUTHTOKEN:
            ngrok.set_auth_token(NGROK_AUTHTOKEN)
            public_url = ngrok.connect(LOCAL_PORT, domain="aghast-dizziness-providing.ngrok-free.dev").public_url
            logging.info("=" * 60)
            logging.info(f"🚀 PUBLIC DASHBOARD URL: {public_url}")
            logging.info("=" * 60)
    except Exception as e:
        logging.error(f"Failed to start Ngrok tunnel: {e}")
    
    await init_db()
    asyncio.create_task(mqtt_listener())
    asyncio.create_task(broadcast_network_health())
    asyncio.create_task(broadcast_network_telemetry())
    yield

    logging.info("Web Dashboard Host shutting down...")
    if public_url:
        try:
            ngrok.disconnect(public_url)
            ngrok.kill()
        except:
            pass


app = FastAPI(lifespan=lifespan)
app.mount("/HTML", StaticFiles(directory="HTML"), name="HTML")

# ---------------------------------------------------------------------------
# Auth helpers & Middleware
# ---------------------------------------------------------------------------
import secrets

async def get_current_user(request: Request):
    token = request.cookies.get("session_token")
    if not token or token not in ACTIVE_SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"},
        )
    
    user_data = ACTIVE_SESSIONS[token]
    
    # If the user is a standard viewer, we must enforce IP Whitelisting
    if user_data["role"] == "viewer":
        client_ip = request.headers.get("x-forwarded-for", request.client.host)
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()
            
        def check_ip_auth(ip):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            # 1. Check if Whitelist is EMPTY (Bootstrap mode)
            c.execute("SELECT COUNT(*) FROM allowed_ips")
            total = c.fetchone()[0]
            if total == 0:
                conn.close()
                return True # Allow all if no IPs are whitelisted yet
            
            # 2. Check if this specific IP is allowed
            c.execute("SELECT 1 FROM allowed_ips WHERE ip_address = ?", (ip,))
            res = c.fetchone()
            conn.close()
            return True if res else False

        try:
            loop = asyncio.get_event_loop()
            allowed = await loop.run_in_executor(None, check_ip_auth, client_ip)
            if not allowed:
                logging.warning(f"[SECURITY] Access Denied for IP: {client_ip}. Please whitelist this IP.")
                raise HTTPException(
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                    headers={"Location": "/access_denied"},
                )
        except HTTPException: raise
        except Exception as e:
            logging.error(f"[AUTH] Security logic error: {e}")
            return user_data # Fail-safe to avoid locking out users on DB error
            
    return user_data

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/login")
async def get_login_page():
    return FileResponse("HTML/login.html")

@app.get("/signup")
async def get_signup_page():
    return FileResponse("HTML/signup.html")
    
@app.get("/access_denied")
async def get_access_denied_page():
    return FileResponse("HTML/access_denied.html")

@app.post("/signup")
async def handle_signup(username: str = Form(...), password: str = Form(...)):
    def do_signup():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if c.fetchone() or username == DUMMY_USER:
            conn.close()
            return False
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, "viewer"))
        conn.commit()
        conn.close()
        return True

    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, do_signup)
    if not success:
        return RedirectResponse(url="/signup?error=exists", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url="/login?success=created", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/login")
async def handle_login(response: Response, username: str = Form(...), password: str = Form(...)):
    # Superuser Check
    if username == DUMMY_USER and password == DUMMY_PASSWORD:
        token = secrets.token_hex(16)
        ACTIVE_SESSIONS[token] = {"username": "admin", "role": "admin"}
        response = RedirectResponse(url="/launcher", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="session_token", value=token, httponly=True)
        return response
        
    # SQLite User Check
    def do_login():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username = ? AND password = ?", (username, password))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    loop = asyncio.get_event_loop()
    role = await loop.run_in_executor(None, do_login)
    if role:
        token = secrets.token_hex(16)
        ACTIVE_SESSIONS[token] = {"username": username, "role": role}
        response = RedirectResponse(url="/launcher", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="session_token", value=token, httponly=True)
        return response
            
    return RedirectResponse(url="/login?error=1", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/api/auth/me")
async def get_me(user_data: dict = Depends(get_current_user)):
    return user_data

# --- IP Management API (Admin Only) ---
@app.get("/api/ips")
async def get_ips(user_data: dict = Depends(get_current_user)):
    if user_data["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    
    def fetch_ips():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS allowed_ips (ip_address TEXT PRIMARY KEY, description TEXT)")
        c.execute("SELECT ip_address FROM allowed_ips")
        rows = c.fetchall()
        conn.close()
        return [{"ip": r[0]} for r in rows]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_ips)

@app.post("/api/ips")
async def add_ip(payload: dict, user_data: dict = Depends(get_current_user)):
    if user_data["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    
    ip_addr = payload.get("ip_address")
    if not ip_addr:
        return {"status": "error", "message": "No IP provided"}
        
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        # Ensure table exists just in case
        c.execute("CREATE TABLE IF NOT EXISTS allowed_ips (ip_address TEXT PRIMARY KEY, description TEXT)")
        c.execute("INSERT OR REPLACE INTO allowed_ips (ip_address, description) VALUES (?, ?)", (ip_addr, ""))
        conn.commit()
        conn.close()
        logging.info(f"[AUTH] Whitelisted new IP: {ip_addr}")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"[API] Error adding IP: {e}")
        return {"status": "error", "message": str(e)}

@app.delete("/api/ips/{ip}")
async def remove_ip(ip: str, user_data: dict = Depends(get_current_user)):
    if user_data["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute("DELETE FROM allowed_ips WHERE ip_address = ?", (ip,))
            await db.commit()
    except Exception as e:
        logging.error(f"[API] Error removing IP: {e}")
    return {"status": "ok"}


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="session_token")
    return response

@app.get("/launcher")
async def get_launcher_page(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/launcher.html")

# ---------------------------------------------------------------------------
# System Launcher API
# ---------------------------------------------------------------------------
@app.post("/api/system/start/{service}")
async def start_service(service: str, user_token: str = Depends(get_current_user)):
    if service not in ACTIVE_PROCESSES:
        raise HTTPException(status_code=400, detail="Invalid service")
    
    if ACTIVE_PROCESSES[service] is not None and ACTIVE_PROCESSES[service].poll() is None:
        return {"status": "already_running"}
        
    try:
        working_dir = os.path.dirname(os.path.abspath(__file__))
        # Very strict venv detection
        venv_python = os.path.join(working_dir, "venv", "bin", "python")
        if not os.path.exists(venv_python):
            venv_python = os.path.join(working_dir, "venv", "bin", "python3")
        if not os.path.exists(venv_python):
            venv_python = sys.executable

        logging.info(f"[DEBUG] Launching {service} using: {venv_python}")
        log_file = os.path.join(working_dir, f"{service}_output.log")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = working_dir
        
        if service == "ai":
            script_path = os.path.join(working_dir, "raspberry_pi", "Server_Pi_AI.py")
            with open(log_file, "w") as f:
                ACTIVE_PROCESSES["ai"] = subprocess.Popen(
                    [venv_python, script_path],
                    cwd=working_dir,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
        elif service == "network":
            script_path = os.path.join(working_dir, "raspberry_pi", "network_fallback.py")
            with open(log_file, "w") as f:
                ACTIVE_PROCESSES["network"] = subprocess.Popen(
                    [venv_python, script_path],
                    cwd=working_dir,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
            
        return {"status": "started", "log": log_file}
    except Exception as e:
        logging.error(f"[LAUNCHER] FATAL ERROR during Root Cause Fix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/stop/{service}")
async def stop_service(service: str, user_token: str = Depends(get_current_user)):
    if service not in ACTIVE_PROCESSES:
        raise HTTPException(status_code=400, detail="Invalid service")
        
    proc = ACTIVE_PROCESSES[service]
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        ACTIVE_PROCESSES[service] = None
        return {"status": "stopped"}
        
    return {"status": "not_running"}

def is_port_open(port):
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(('127.0.0.1', port)) == 0
    except:
        return False

@app.get("/api/system/status")
async def system_status(user_token: str = Depends(get_current_user)):
    # AI Engine uses 8001, Network Fallback uses 8003 (assumed)
    ai_status = "running" if is_port_open(8001) else "stopped"
    net_status = "running" if is_port_open(8003) else "stopped"
    
    # Also check our tracked processes
    for svc, proc in ACTIVE_PROCESSES.items():
        if proc is not None and proc.poll() is None:
            if svc == "ai": ai_status = "running"
            if svc == "network": net_status = "running"
            
    return {"ai": ai_status, "network": net_status}

@app.get("/")
async def get_homepage(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/index.html")


@app.get("/network")
async def get_network_page(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/network.html")


@app.get("/records")
async def get_records_page(user_token: str = Depends(get_current_user)):
    return FileResponse("HTML/history.html")


@app.get("/api/video")
async def video_proxy():
    """Proxies MJPEG stream from AI Server (8001) to Dashboard."""
    # We use localhost because both servers run on the Pi
    PI_VIDEO_URL = "http://localhost:8001/video_feed"
    
    async def stream_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("GET", PI_VIDEO_URL) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            except Exception as e:
                logging.error(f"[VIDEO PROXY] Failed to connect to AI Server: {e}")

    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/patients", response_model=List[PatientListItem])
async def get_all_patients(user_token: str = Depends(get_current_user)):
    def fetch_patients():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT DISTINCT patient_id, patient_name FROM patient_records")
        rows = c.fetchall()
        conn.close()
        return [{"_id": r[0], "patient_name": r[1]} for r in rows]

    patient_list = []
    try:
        loop = asyncio.get_event_loop()
        patient_list = await loop.run_in_executor(None, fetch_patients)
    except Exception as e:
        logging.error(f"[API] Error fetching patients: {e}")
    return patient_list


@app.get("/api/history_all")
async def get_all_patient_records(user_token: str = Depends(get_current_user)):
    def fetch_history():
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM patient_records ORDER BY id DESC LIMIT 1000")
        rows = c.fetchall()
        conn.close()
        
        data = []
        for row in rows:
            raw_ts = row["timestamp"]
            if raw_ts and not raw_ts.endswith("Z"):
                iso_ts = raw_ts + "Z"
            else:
                iso_ts = raw_ts or "0000-00-00T00:00:00Z"
            
            data.append({
                "patient_id":        row["patient_id"],
                "patient_name":      row["patient_name"],
                "heart_rate":        row["hr"],
                "oxygen":            row["spo2"],
                "temperature":       row["temp"],
                "emotion":           row["emotion"],
                "fall_status":       row["fall_status"],
                "stage":             row["stage"],
                "_server_timestamp": iso_ts,
            })
        return data

    all_data = []
    try:
        loop = asyncio.get_event_loop()
        all_data = await loop.run_in_executor(None, fetch_history)
    except Exception as e:
        logging.error(f"[API] History error: {e}")
        
    return {"records": all_data}

@app.get("/api/generate_report/{patient_id}")
async def generate_pdf_report(patient_id: str):
    # --- NEW SIMPLIFIED GENERATOR ---
    records = []
    patient_name = "Fernando" # Default for Viva
    
    # 1. Direct Sync Fetch (More stable for reports)
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM patient_records WHERE patient_id = ? ORDER BY id DESC LIMIT 100", (patient_id,))
        rows = c.fetchall()
        for r in rows:
            if r["patient_name"]: patient_name = r["patient_name"]
            records.append(dict(r))
        conn.close()
    except Exception as e:
        logging.error(f"[REWRITE] DB Error: {e}")

    # 2. Analytics
    total_falls = 0
    in_fall = False
    hr_sum, spo2_sum, valid_count = 0, 0, 0
    
    for r in reversed(records):
        # Incident Counting
        is_fall = "FALL" in str(r.get("fall_status", "")).upper()
        if is_fall and not in_fall:
            total_falls += 1
            in_fall = True
        elif not is_fall:
            in_fall = False
            
        hr = r.get("hr", 0)
        if 40 <= hr <= 180:
            hr_sum += hr
            spo2_sum += r.get("spo2", 0)
            valid_count += 1

    avg_hr = round(hr_sum/valid_count, 1) if valid_count > 0 else 0
    avg_spo2 = round(spo2_sum/valid_count, 1) if valid_count > 0 else 0

    # 3. PDF Creation
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Header
    title = f"ELDERHUB: Health Analytics Report - {patient_name}"
    elements.append(Paragraph(title, styles['Heading1']))
    elements.append(Paragraph(f"Analysis Period: Last 100 Vital Sign Samples", styles['Normal']))
    elements.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Metrics Table
    data = [
        ["HEALTH METRIC", "MEASURED VALUE"],
        ["Average Heart Rate", f"{avg_hr} BPM"],
        ["Average Oxygen (SpO2)", f"{avg_spo2} %"],
        ["Total Fall Incidents", f"{total_falls}"],
        ["Records Analyzed", f"{len(records)}"]
    ]
    
    t = Table(data, colWidths=[250, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("End of Automated Health Summary", styles['Italic']))
    
    doc.build(elements)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={'Content-Disposition': f'attachment; filename="Vital_Report.pdf"'})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.cookies.get("session_token")
    if not token or token not in ACTIVE_SESSIONS:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logging.warning("WebSocket connection rejected: No valid session token.")
        return
    await manager.connect(websocket)
    logging.info("Browser dashboard client connected. Hydrating recent state...")
    
    # Safely hydrate the dashboard with the last 15 events to prevent empty layout
    def fetch_recent():
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM patient_records WHERE patient_id = 'P001' ORDER BY id DESC LIMIT 15")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    try:
        loop = asyncio.get_event_loop()
        recent_rows = await loop.run_in_executor(None, fetch_recent)
        for row in reversed(recent_rows):
            try:
                await websocket.send_text(json.dumps(row))
                await asyncio.sleep(0.05) 
            except Exception:
                pass
    except Exception as e:
        logging.warning(f"State hydration failed: {e}")

    try:
        while True:
            msg_text = await websocket.receive_text()
            try:
                msg = json.loads(msg_text)
                if msg.get("type") == "request_sync":
                    logging.info("[WS] Client requested manual sync. Triggering broadcast...")
                    pass 
                elif msg.get("type") == "manual_fall_trigger":
                    logging.info("[WS] MANUAL FALL TRIGGER RECEIVED (F-Key). Broadcasting to Pi...")
                    # Send to MQTT so the Pi AI Engine can trigger the hardware buzzer
                    async with aiomqtt.Client(hostname=MQTT_BROKER_IP, port=MQTT_PORT) as client:
                        await client.publish("r25_014/vitals/raw", payload=json.dumps({"pid":"P001", "manual_fall": True}))
            except:
                pass
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
