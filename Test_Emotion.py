import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt

mapper = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

with open("Emotional/model.yaml", "r") as json_file:
    model_json = json_file.read()

with tf.keras.utils.custom_object_scope({'Functional': tf.keras.Model}):
    loaded_model = model_from_json(model_json)

loaded_model.load_weights("Emotional/model.h5")

def predict_emotion(image_path, model, label_mapper):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: Could not load image from {image_path}")
        return
    img_resized = cv2.resize(img_gray, (48, 48), interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    img_final = np.expand_dims(img_rgb, axis=0)
    predictions = model.predict(img_final)
    predicted_index = np.argmax(predictions[0])
    predicted_emotion = label_mapper.get(predicted_index, "Unknown Emotion")
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Predicted Emotion: {predicted_emotion}")
    plt.axis('off')
    plt.show()
    return predicted_emotion

image_file = "Test_Images/10018.png"
predicted_label = predict_emotion(image_file, loaded_model, mapper)
if predicted_label:
    print(f"Final Prediction: {predicted_label}")