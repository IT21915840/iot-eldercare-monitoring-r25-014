#!/usr/bin/env bash
# =============================================================================
#  setup.sh  —  Patient Monitoring System: Raspberry Pi Setup Script
#  Run once after copying this folder to your Pi:
#    chmod +x setup.sh && sudo ./setup.sh
# =============================================================================
set -e

echo ""
echo "=============================================="
echo "  Patient Monitor — Raspberry Pi Setup"
echo "=============================================="
echo ""

# ── 1. System update & dependencies ──────────────────────────────────────────
echo "[1/5] Updating system and installing system packages ..."
apt-get update -y
apt-get install -y \
    python3 python3-pip python3-venv \
    mosquitto mosquitto-clients \
    libopenblas-dev libhdf5-dev \
    libopencv-dev \
    git curl

# ── 2. Enable & start Mosquitto MQTT broker ───────────────────────────────────
echo "[2/5] Enabling Mosquitto MQTT broker ..."
systemctl enable mosquitto
systemctl start mosquitto
echo "      Mosquitto status: $(systemctl is-active mosquitto)"

# ── 3. MongoDB (optional — stores patient records) ───────────────────────────
echo "[3/5] Installing MongoDB ..."
# Import MongoDB public GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
    gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add MongoDB repo for Raspberry Pi (Debian bookworm arm64)
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
https://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" \
    | tee /etc/apt/sources.list.d/mongodb-org-7.0.list

apt-get update -y
apt-get install -y mongodb-org || echo "  [WARN] MongoDB install failed — data will not be saved, dashboard still works."

systemctl enable mongod  2>/dev/null || true
systemctl start  mongod  2>/dev/null || true

# ── 4. Python virtual environment & pip packages ─────────────────────────────
echo "[4/5] Setting up Python virtual environment ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install \
    fastapi \
    uvicorn \
    motor \
    pymongo \
    pydantic \
    pytz \
    websockets \
    python-multipart \
    aiomqtt \
    joblib \
    pandas \
    scikit-learn \
    opencv-python-headless \
    tensorflow \
    numpy

deactivate

# ── 5. Make run script executable ────────────────────────────────────────────
echo "[5/5] Making run.sh executable ..."
chmod +x "$SCRIPT_DIR/run.sh"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Copy your AI model files into:"
echo "       - Emotional/model.yaml  +  model.h5"
echo "       - Fall_detection/keras_Model.h5  +  labels.txt"
echo "       - Sleep/sleep_stage_model.joblib  +  scaler.joblib"
echo ""
echo "    2. Set your ESP32's MQTT broker IP to this Pi's IP"
echo "       (shows in captive portal on ESP32 first boot)"
echo ""
echo "    3. Run the system:"
echo "       ./run.sh"
echo "=============================================="
