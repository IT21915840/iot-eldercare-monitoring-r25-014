# ElderHub | Eldercare Intelligence System

Welcome to the **ElderHub** project. This system is designed for real-time health monitoring, fall detection, and emotion analysis for elderly care.

This guide will help you set up your local environment to contribute to the project.

---

## 🛠️ Prerequisites

Before you start, ensure you have the following installed on your laptop:

1.  **Python 3.10+**: [Download](https://www.python.org/downloads/)
2.  **Git**: [Download](https://git-scm.com/downloads)
3.  **MongoDB (Community Server)**: [Download](https://www.mongodb.com/try/download/community)
    -   This is the database engine. Ensure it is installed and running on port `27017`.
    -   *Tip: During installation, check "Install MongoDB Compass" to get a visual dashboard for your data.*
4.  **Mosquitto MQTT Broker**: [Download](https://mosquitto.org/download/)
    -   The system relies on an MQTT broker. Ensure Mosquitto is running on `localhost:1883`.

---

## 🗂️ Setup Instructions

### 1. Clone the Project
Open your terminal, navigate to where you want the project to live (e.g., your Documents folder), and run:
```bash
git clone https://github.com/IT21915840/iot-eldercare-monitoring-r25-014.git
cd iot-eldercare-monitoring-r25-014
git checkout network  # Switch to the network branch
```
*Note: This will create a folder named `iot-eldercare-monitoring-r25-014` in your current directory.*

### 2. Install Dependencies
Create a virtual environment and install the required libraries:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. ⚠️ Manual File Transfer (CRITICAL)
Due to size limits, the following AI model files are **not** in the Git repository. You must manually copy these from the project owner and place them in these exact folders:

| File Name | Destination Folder (within project root) |
| :--- | :--- |
| `model.h5` | `/Emotional/` |
| `emotion_model.tflite` | `/Emotional/` |
| `keras_Model.h5` | `/Fall_detection/` |
| `fall_model.tflite` | `/Fall_detection/` |
| `labels.txt` | `/Fall_detection/` |

Ensure these are placed in their respective folders before running the AI server.

---

## 🚀 Running the Project

To see the system in action, you need to run three separate components in three different terminal windows:

### Terminal 1: The Main Web & Data Server
This handles the dashboard UI and database storage.
```bash
python Server_Main.py
```
*Accessible at: `http://localhost:8002`*

### Terminal 2: The AI Vision Engine
This handles camera processing, emotion detection, and fall alerts.
```bash
cd raspberry_pi
python Server_Pi_AI.py
```

### Terminal 3: Mock Sensor Data (For Testing)
If you don't have the ESP32 hardware, run this to simulate heart rate, SpO2, and temperature data:
```bash
python mock_esp32.py
```

---

## ⚙️ Configuration
If you need to change the server ports or database URLs, check the top of these files:
- **Web Server/DB**: `Server_Main.py`
- **AI/MQTT**: `raspberry_pi/Server_Pi_AI.py`

---

## 🆘 Troubleshooting
- **MQTT Connection Error**: Ensure your Mosquitto service is started.
- **MongoDB Error**: Ensure the MongoDB service is active. You can check this using `mongosh` or MongoDB Compass.
---

## 🍓 Running on a Raspberry Pi
If you are running the AI Vision engine on a Raspberry Pi and the Dashboard on your Laptop:
1.  **Find your Laptop IP**: Run `ipconfig` (Windows) or `ifconfig` (Mac/Linux) on your laptop to find your local IP (e.g., `192.168.1.10`).
2.  **Update Config**: In `raspberry_pi/Server_Pi_AI.py`, change `MQTT_BROKER_HOST = "localhost"` to your laptop's IP address.
3.  **Ensure Same Network**: Both the Pi and the Laptop must be connected to the same WiFi/Network.
4.  **Remote Access**: We highly recommend using the VS Code **Remote - SSH** extension. It allows you to edit code on the Pi directly from your laptop without needing a separate monitor for the Pi.

