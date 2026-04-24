#!/usr/bin/env python3
"""
convert_models.py  —  Run this ONCE on your PC (not Pi) to convert
Keras .h5 models → .tflite format for use with tflite-runtime on the Pi.

Requirements on PC:
    pip install tensorflow

Usage:
    python convert_models.py
"""

import os

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: tensorflow not installed. Run:  pip install tensorflow")
    raise SystemExit(1)


def convert(h5_path: str, tflite_path: str):
    if not os.path.exists(h5_path):
        print(f"  SKIP — not found: {h5_path}")
        return
    print(f"  Converting {h5_path} → {tflite_path} ...")
    model      = tf.keras.models.load_model(h5_path, compile=False)
    converter  = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_buf = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_buf)
    print(f"  Done. Size: {len(tflite_buf) / 1024:.1f} KB")


if __name__ == "__main__":
    print("=== Model Converter: Keras H5 → TFLite ===\n")

    convert(
        h5_path    ="Emotional/model.h5",
        tflite_path="Emotional/emotion_model.tflite",
    )
    convert(
        h5_path    ="Fall_detection/keras_Model.h5",
        tflite_path="Fall_detection/fall_model.tflite",
    )

    print("\nConversion complete!")
    print("Copy the .tflite files alongside your Pi scripts.")
