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
    print(f"  Converting {h5_path} to {tflite_path} ...")
    try:
        model      = tf.keras.models.load_model(h5_path, compile=False)
    except Exception as e:
        yaml_path = h5_path.replace(".h5", ".yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                json_str = f.read()
            with tf.keras.utils.custom_object_scope({'Functional': tf.keras.Model}):
                model = tf.keras.models.model_from_json(json_str)
            model.load_weights(h5_path)
        else:
            raise e

    converter  = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_buf = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_buf)
    print(f"  Done. Size: {len(tflite_buf) / 1024:.1f} KB")


if __name__ == "__main__":
    print("=== Model Converter: Keras H5 to TFLite ===\n")

    # convert(
    #     h5_path    ="Emotional/model.h5",
    #     tflite_path="Emotional/emotion_model.tflite",
    # )
    convert(
        h5_path    ="Fall_detection/keras_Model.h5",
        tflite_path="Fall_detection/fall_model.tflite",
    )

    print("\nConversion complete!")
    print("Copy the .tflite files alongside your Pi scripts.")
