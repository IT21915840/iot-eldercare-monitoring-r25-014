import asyncio
import websockets
import json
import logging
import datetime
import joblib
import pandas as pd
import warnings
from sklearn.base import InconsistentVersionWarning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning,
    module="sklearn"
)

ESP32_IP = "172.20.10.3"
WEBSOCKET_URL = f"ws://{ESP32_IP}/ws"

try:
    loaded_model = joblib.load('Sleep/sleep_stage_model.joblib')
    loaded_scaler = joblib.load('Sleep/scaler.joblib')
    logging.info("ML model and scaler loaded successfully.")
except FileNotFoundError as e:
    logging.error(
        f"Error loading model files. Please ensure 'Sleep/sleep_stage_model.joblib' and 'Sleep/scaler.joblib' exist. Details: {e}")


def predict_sleep_stage(spo2: float, hr: float, temp: float) -> str:
    if 'loaded_model' not in globals() or 'loaded_scaler' not in globals():
        return "MODEL_MISSING"

    try:
        feature_names = ['spo2', 'hr', 'temp']
        sample_data = pd.DataFrame([[spo2, hr, temp]], columns=feature_names)

        sample_data_scaled_array = loaded_scaler.transform(sample_data)

        sample_data_scaled = pd.DataFrame(
            sample_data_scaled_array,
            columns=feature_names
        )

        predicted_sleep_stage = loaded_model.predict(sample_data_scaled)
        return predicted_sleep_stage[0]

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return "PRED_FAIL"


async def receive_data():
    logging.info(f"Attempting to connect to: {WEBSOCKET_URL}")

    while True:
        try:
            async with websockets.connect(WEBSOCKET_URL) as websocket:
                logging.info("Connection established. Waiting for data...")

                async for message in websocket:
                    try:
                        data = json.loads(message)

                        hr = data.get('hr', 0)
                        spo2 = data.get('spo2', 0)
                        temp = data.get('temp', 0.0)

                        stage = predict_sleep_stage(spo2, hr, temp)

                        current_time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                        yield {
                            'time': current_time_stamp,
                            'hr': hr,
                            'spo2': spo2,
                            'temp': temp,
                            'stage': stage
                        }

                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON: {message[:50]}...")
                    except Exception as e:
                        logging.error(f"Error processing message: {e}")

        except ConnectionRefusedError:
            logging.error(f"Connection refused by {ESP32_IP}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except websockets.exceptions.InvalidURI:
            logging.error("Invalid WebSocket URL. Please check the IP address.")
            return
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)


async def main():
    print("\n" + "=" * 87)
    print(f"| {'TIMESTAMP':<24} | {'HR (BPM)':<10} | {'SpO2 (%)':<10} | {'TEMP (°C)':<10} | {'SLEEP STAGE':<15} |")
    print("=" * 87)

    async for data_point in receive_data():
        print(
            f"| {data_point['time']:<24} | {data_point['hr']:<10} | {data_point['spo2']:<10} | {data_point['temp']:<10.2f} | {data_point['stage']:<15} |")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Client closed by user.")
    except Exception as e:
        logging.error(f"Main script error: {e}")