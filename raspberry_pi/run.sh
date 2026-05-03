#!/usr/bin/env bash
# =============================================================================
#  run.sh  —  Patient Monitoring System: Start All Services
#  Usage:  ./run.sh
#  Stop:   Press Ctrl+C  (kills all background services cleanly)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin/python"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
CYAN="\033[0;36m"
RED="\033[0;31m"
NC="\033[0m"

log()  { echo -e "${CYAN}[RUN]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK]${NC} $*"; }
fail() { echo -e "${RED}[ERR]${NC} $*"; }

echo ""
echo "=============================================="
echo "  Patient Monitor — Raspberry Pi Launcher"
echo "=============================================="
echo ""

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    log "Shutting down all services ..."
    kill "$PID_AI"  2>/dev/null && ok  "Server_Pi_AI stopped"  || true
    kill "$PID_WEB" 2>/dev/null && ok  "Server_Web stopped"    || true
    wait 2>/dev/null
    echo "Done. Goodbye."
}
trap cleanup INT TERM

# ── 1. Mosquitto MQTT broker ──────────────────────────────────────────────────
log "Checking Mosquitto MQTT broker ..."
if systemctl is-active --quiet mosquitto 2>/dev/null; then
    ok "Mosquitto already running."
else
    log "Starting Mosquitto ..."
    mosquitto -d 2>/dev/null || {
        fail "Could not start Mosquitto. Run:  sudo systemctl start mosquitto"
        exit 1
    }
    sleep 1
    ok "Mosquitto started."
fi

# ── 2. MongoDB ────────────────────────────────────────────────────────────────
log "Checking MongoDB ..."
if systemctl is-active --quiet mongod 2>/dev/null; then
    ok "MongoDB already running."
else
    log "Trying to start MongoDB ..."
    systemctl start mongod 2>/dev/null || \
    mongod --dbpath /data/db --fork --logpath /tmp/mongod.log 2>/dev/null || \
    log "MongoDB unavailable — records won't be saved (dashboard still works)."
fi

# ── 3. Python venv check ─────────────────────────────────────────────────────
if [ ! -f "$VENV" ]; then
    fail "Virtual environment not found. Run setup.sh first."
    exit 1
fi

# ── 4. Move to root for Server_Main paths ─────────────────────────────────────
cd "$SCRIPT_DIR/.."

# ── 5. Web Dashboard Server (port 8002) ───────────────────────────────────────
log "Starting Server_Main.py on port 8002 ..."
"$VENV" "Server_Main.py" > /tmp/pi_web.log 2>&1 &
PID_WEB=$!
sleep 2
if kill -0 "$PID_WEB" 2>/dev/null; then
    ok "Server_Main running    (PID $PID_WEB) — logs: /tmp/pi_web.log"
else
    fail "Server_Main failed to start. Check /tmp/pi_web.log"
    cat /tmp/pi_web.log
    exit 1
fi

# ── Ready ─────────────────────────────────────────────────────────────────────
PI_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "=============================================="
echo -e "  ${GREEN}All services running!${NC}"
echo ""
echo "  Dashboard : http://${PI_IP}:8002/"
echo "  Login     : admin / password"
echo ""
echo "  AI Server : ws://${PI_IP}:8001/ws"
echo "  MQTT      : ${PI_IP}:1883  (topic: vitals/raw)"
echo ""
echo "  Set your ESP32 MQTT broker IP to: ${PI_IP}"
echo ""
echo "  Press Ctrl+C to stop everything."
echo "=============================================="
echo ""

# Wait forever (until Ctrl+C)
wait
