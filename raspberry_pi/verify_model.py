import os
import cv2
import glob
import json
import logging
import argparse
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FALL_MODEL_TFLITE = os.path.join(BASE_DIR, "Models", "Fall", "fall_model.tflite")
FALL_LABELS_TXT = os.path.join(BASE_DIR, "Models", "Fall", "labels.txt")
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def load_tflite_model(path: str):
    try:
        if not os.path.exists(path):
            logging.error(f"TFLite model not found at {path}")
            return None
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        logging.info("TFLite model loaded successfully.")
        return interpreter
    except Exception as e:
        logging.error(f"Error loading TFLite model: {e}")
        return None

def load_labels() -> list:
    try:
        with open(FALL_LABELS_TXT, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        logging.error(f"Error loading labels: {e}")
        return []

def run_inference(interpreter, input_arr):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_arr)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        logging.error(f"Inference error: {e}")
        return np.array([])

def test_image(image_path: str, hog, face_cascade, interpreter, labels):
    frame = cv2.imread(image_path)
    if frame is None:
        logging.error(f"Could not read {image_path}")
        return None, 0.0, False

    # Check for face fallback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    fallback_detected = len(faces) > 0

    # HOG Detection
    (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    best_box = None
    best_weight = 0.0

    if len(boxes) > 0 and len(weights) > 0:
        for (x, y, w, h), weight in zip(boxes, weights):
            w_val = weight[0] if isinstance(weight, (list, np.ndarray)) else weight
            if w_val > 0.4 and w_val > best_weight:
                best_weight = w_val
                best_box = (x, y, w, h)

    # If no person detected
    if best_box is None and not fallback_detected:
        return "No person detected by HOG/Haar", 0.0, False

    # Isolate Human Figure
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if best_box is not None:
        (x, y, w, h) = best_box
        margin = int(0.1 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        person_crop = rgb_frame[y1:y2, x1:x2]
        used_crop = True
    else:
        person_crop = rgb_frame
        used_crop = False

    # Resize and Normalize for DenseNet
    resized = cv2.resize(person_crop, (224, 224)).astype(np.float32)
    arr = ((resized / 127.5) - 1.0).reshape(1, 224, 224, 3)

    # Inference
    pred = run_inference(interpreter, arr)
    if pred.size == 0:
        return "Inference failed", 0.0, used_crop

    idx = int(np.argmax(pred[0]))
    raw_label = labels[idx] if idx < len(labels) else "Unknown"
    conf = float(pred[0][idx] * 100)

    # Map label
    raw_upper = raw_label.upper()
    if "NOT" in raw_upper or "NO" in raw_upper:
        final_label = "Normal"
    elif "FALL" in raw_upper:
        final_label = "FALL DETECTED"
    else:
        final_label = raw_label

    return final_label, conf, used_crop

def main():
    parser = argparse.ArgumentParser(description="Verify Fall Detection Model Accuracy")
    parser.add_argument("--image", type=str, help="Path to a single image to test")
    parser.add_argument("--dir", type=str, help="Path to a directory of images to test")
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("Please provide an --image or a --dir. Example: python verify_model.py --image sample.jpg")
        return

    # Load resources
    interpreter = load_tflite_model(FALL_MODEL_TFLITE)
    labels = load_labels()
    if not interpreter or not labels:
        return

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("========================================")
    print("      FALL DETECTION VERIFICATION       ")
    print("========================================")

    images_to_test = []
    if args.image:
        images_to_test.append(args.image)
    if args.dir:
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images_to_test.extend(glob.glob(os.path.join(args.dir, ext)))

    if not images_to_test:
        print("No images found.")
        return

    results = {"Falls Detected": 0, "Normal": 0, "No Person": 0, "Errors": 0}

    for img_path in images_to_test:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        label, conf, cropped = test_image(img_path, hog, face_cascade, interpreter, labels)
        
        status = "✅ Successfully Cropped" if cropped else "❌ Used Full Frame (No HOG)"
        print(f"HOG Status : {status}")
        print(f"Prediction : {label} ({conf:.1f}%)")

        if "FALL" in label: results["Falls Detected"] += 1
        elif label == "Normal": results["Normal"] += 1
        elif "No person" in label: results["No Person"] += 1
        else: results["Errors"] += 1

    print("\n========================================")
    print("             FINAL SUMMARY              ")
    print("========================================")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
