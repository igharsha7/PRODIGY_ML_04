import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model_loader import predict_gesture
import os

gestures = ['Palm', 'L', 'Fist', 'Fist Moved', 'Thumb', 'Index', 'OK', 'Palm Moved', 'C', 'Down']

st.title('Hand Gesture Recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    predicted_class, prediction = predict_gesture(image)
    
    st.write(f'Predicted Gesture: {gestures[predicted_class]}')
    st.write(f'Prediction Confidence: {np.max(prediction) * 100:.2f}%')

    st.write("Prediction Probabilities:")
    for i, gesture in enumerate(gestures):
        st.write(f'{gesture}: {prediction[0][i] * 100:.2f}%')


example_gesture = st.selectbox('Select an example gesture:', gestures)
if st.button('Show Example'):
    example_folder = f'E:\\Prodigy ML\\Task04\\leapGestRecog\\{str(gestures.index(example_gesture)).zfill(2)}'
    subfolders = os.listdir(example_folder)
    if subfolders:
        example_subfolder = os.path.join(example_folder, subfolders[0])
        example_images = os.listdir(example_subfolder)
        if example_images:
            example_image_path = os.path.join(example_subfolder, example_images[0])
            example_image = Image.open(example_image_path)
            st.image(example_image, caption=f'Example: {example_gesture}', use_column_width=True)
        else:
            st.write("No example images found.")
    else:
        st.write("No example subfolders found.")
