#!/usr/bin/env bash
# =============================================================================
#  setup.sh  —  Patient Monitoring System: Raspberry Pi Setup Script
#  Run once after copying this folder to your Pi:
#    chmod +x setup.sh && ./setup.sh
# =============================================================================
set -e

# Support for sudo detection
SUDO=""
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
fi

echo ""
echo "=============================================="
echo "  Patient Monitor — Raspberry Pi Setup"
echo "=============================================="
echo ""

# ── 1. System update & dependencies ──────────────────────────────────────────
echo "[1/5] Updating system and installing system packages ..."

# Cleanup any broken MongoDB list from previous attempts
$SUDO rm -f /etc/apt/sources.list.d/mongodb-org-7.0.list || true

# Temporarily disable exit-on-error for the system update
set +e
$SUDO apt-get update -y
$SUDO apt-get install -y \
    python3 python3-pip python3-venv \
    mosquitto mosquitto-clients \
    libopenblas-dev libhdf5-dev \
    libgl1 libglib2.0-0 \
    git curl gpg
set -e

# ── 2. Enable & start Mosquitto MQTT broker ───────────────────────────────────
echo "[2/5] Enabling Mosquitto MQTT broker ..."
$SUDO systemctl enable mosquitto || true
$SUDO systemctl start mosquitto || true
echo "      Mosquitto status: $(systemctl is-active mosquitto || echo 'inactive')"

# ── 3. MongoDB (optional) ───────────────────────────────────────────────────
echo "[3/5] Installing MongoDB (Optional) ..."
# We use a subshell to strictly isolate the repo failure
(
    set +e
    # Try to fix GPG but don't fail if it doesn't work
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
        $SUDO gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg --yes

    # Add MongoDB repo (using Bookworm for best compatibility on Trixie)
    echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
https://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" \
        | $SUDO tee /etc/apt/sources.list.d/mongodb-org-7.0.list

    $SUDO apt-get update -y
    $SUDO apt-get install -y mongodb-org
) || echo "  [SKIPPED] MongoDB install failed (Repo blocked) — database features will be disabled."

$SUDO systemctl enable mongod 2>/dev/null || true
$SUDO systemctl start  mongod 2>/dev/null || true

# ── 4. Python virtual environment & pip packages ─────────────────────────────
echo "[4/5] Setting up Python virtual environment ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure old .venv is gone if it's broken
rm -rf .venv

python3 -m venv .venv
source .venv/bin/activate

echo "Installing Python packages (this may take 5-10 minutes) ..."
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
    numpy \
    httpx

deactivate

# ── 5. Permissions ────────────────────────────────────────────────────────────
echo "[5/5] Finalizing permissions ..."
chmod +x "$SCRIPT_DIR/run.sh"

# Fix ownership if script was run with sudo
if [ ! -z "$SUDO_USER" ]; then
    chown -R "$SUDO_USER:$SUDO_USER" .venv
fi

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Run the system with:"
echo "    ./run.sh"
echo "=============================================="
