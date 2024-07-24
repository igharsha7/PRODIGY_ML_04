import tensorflow as tf
import keras
from keras import models
import os

model_path = os.path.abspath('hand_gesture_model.keras')
model1 = keras.models.load_model(model_path)

def preprocess_image(image):
    import cv2
    # If the image has 4 channels (RGBA), this converts it to RGB first
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # If the image is in color, this converts it to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    image = image.reshape(1, 64, 64, 1)
    return image

def predict_gesture(image):
    processed_image = preprocess_image(image)
    prediction = model1.predict(processed_image)
    predicted_class = prediction.argmax()
    return predicted_class, prediction