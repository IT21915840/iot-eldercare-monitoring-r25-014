import time
import subprocess
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [FALLBACK] %(message)s")

# Configuration
PRIMARY_SSID = "Theneth"
PRIMARY_PASS = "qwertyuio"

SECONDARY_SSID = "Backup_SSID"
SECONDARY_PASS = "12345678"


PING_TARGET = "8.8.8.8"
CHECK_INTERVAL = 10  # Seconds
MAX_FAILURES = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "fallback_state.json")

def write_state(active_network):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"active_network": active_network}, f)
    except Exception as e:
        logging.error(f"Failed to write state: {e}")

def ping_check():
    # Ping once, wait 1 second max
    cmd = ["ping", "-n", "1", "-w", "1000", PING_TARGET] if os.name == 'nt' else ["ping", "-c", "1", "-W", "1", PING_TARGET]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False

def switch_network(ssid, password):
    logging.info(f"Attempting to connect to {ssid}...")
    try:
        if os.name == 'nt':
            # Windows fallback (for testing locally)
            cmd = f'netsh wlan connect name="{ssid}"'
        else:
            # Linux / Raspberry Pi fallback using nmcli
            cmd = f'nmcli dev wifi connect "{ssid}" password "{password}"'
            
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logging.info(f"Successfully connected to {ssid}")
            return True
        else:
            logging.error(f"Failed to connect to {ssid}: {result.stderr.decode('utf-8')}")
            return False
    except Exception as e:
        logging.error(f"Network switch exception: {e}")
        return False

def get_active_ssid():
    try:
        if os.name == 'nt':
            result = subprocess.run('netsh wlan show interfaces', capture_output=True, text=True, shell=True)
            for line in result.stdout.split('\n'):
                if "SSID" in line and "BSSID" not in line:
                    return line.split(":")[1].strip()
        else:
            result = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            
            # Fallback for newer NetworkManager based OS
            result = subprocess.run("nmcli -t -f active,ssid dev wifi | grep '^yes:' | cut -d: -f2", shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except Exception:
        pass
    return ""

def main():
    logging.info("Network Fallback Daemon Started.")
    
    active_ssid = get_active_ssid()
    if active_ssid == SECONDARY_SSID:
        current_network = SECONDARY_SSID
        write_state(f"Secondary ({SECONDARY_SSID})")
        logging.info(f"Detected startup network: {SECONDARY_SSID} (Secondary)")
    else:
        current_network = PRIMARY_SSID
        write_state(f"Primary ({PRIMARY_SSID})")
        logging.info(f"Detected startup network: {PRIMARY_SSID} (Primary)")
        
    failures = 0
    
    while True:
        # Periodic SSID sync for Dashboard
        new_ssid = get_active_ssid()
        if new_ssid != active_ssid:
            active_ssid = new_ssid
            label = "Primary" if active_ssid == PRIMARY_SSID else "Secondary" if active_ssid == SECONDARY_SSID else "Unknown"
            write_state(f"{label} ({active_ssid})")
            logging.info(f"SSID changed detected: {active_ssid}")

        if ping_check():
            failures = 0
            if current_network == SECONDARY_SSID:
                pass
        else:
            failures += 1
            logging.warning(f"Ping failed ({failures}/{MAX_FAILURES})")
            
            if failures >= MAX_FAILURES:
                logging.error("Network connection lost. Initiating fallback sequence...")
                
                if current_network == PRIMARY_SSID:
                    success = switch_network(SECONDARY_SSID, SECONDARY_PASS)
                    if success:
                        current_network = SECONDARY_SSID
                        write_state(f"Secondary ({SECONDARY_SSID})")
                        failures = 0
                else:
                    success = switch_network(PRIMARY_SSID, PRIMARY_PASS)
                    if success:
                        current_network = PRIMARY_SSID
                        write_state(f"Primary ({PRIMARY_SSID})")
                        failures = 0
                        
                time.sleep(10)
                
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

