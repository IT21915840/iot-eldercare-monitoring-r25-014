import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import random
import aiomqtt

# Mock ESP32 Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC = "vitals/raw"

async def mock_esp32():
    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        async with aiomqtt.Client(hostname=MQTT_BROKER, port=MQTT_PORT) as client:
            print("Connected! Starting to send mock ESP32 data...")
            
            profiles = [
                {"stage": "Awake", "hr_range": (70, 90), "spo2_range": (96, 100), "temp_range": (36.5, 37.5)},
                {"stage": "Light", "hr_range": (60, 75), "spo2_range": (95, 98), "temp_range": (36.0, 36.5)},
                {"stage": "Deep", "hr_range": (50, 65), "spo2_range": (94, 96), "temp_range": (35.5, 36.0)},
                {"stage": "REM", "hr_range": (65, 85), "spo2_range": (95, 98), "temp_range": (36.0, 36.8)}
            ]
            cycle_counter = 0

            while True:
                # Rotate through profiles every 5 iterations (10 seconds)
                profile = profiles[(cycle_counter // 5) % len(profiles)]
                cycle_counter += 1

                hr = random.randint(*profile["hr_range"])
                spo2 = random.randint(*profile["spo2_range"])
                temp = round(random.uniform(*profile["temp_range"]), 1)
                crit = 0
                
                # Simulate a critical condition frequently (30% chance)
                if random.random() < 0.30:
                    crit = 1
                    hr = random.randint(110, 150)
                    spo2 = random.randint(80, 89)
                    temp = round(random.uniform(38.0, 40.0), 1)

                payload = {
                    "pid": "P001",
                    "hr": hr,
                    "spo2": spo2,
                    "temp": temp,
                    "crit": crit
                }
                
                print(f"Publishing to '{TOPIC}': {payload}")
                await client.publish(TOPIC, payload=json.dumps(payload))
                
                # Send data every 2 seconds
                await asyncio.sleep(2)

    except aiomqtt.MqttError as e:
        print(f"MQTT Error: {e}")
        print("-> Make sure you have an MQTT broker (like Mosquitto) running on localhost:1883!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(mock_esp32())
    except KeyboardInterrupt:
        print("\nMock ESP32 Stopped.")
