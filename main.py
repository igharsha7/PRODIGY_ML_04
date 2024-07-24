import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



dataset_path = 'E:\\Prodigy ML\\Task04\\leapGestRecog'
gestures = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']


data = []
labels = []


image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

for i, gesture in enumerate(gestures):
    gesture_folder = os.path.join(dataset_path, gesture)
    print(f"Processing gesture folder: {gesture_folder}")
    for subfolder in os.listdir(gesture_folder):
        subfolder_path = os.path.join(gesture_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for img_name in os.listdir(subfolder_path):
                if any(img_name.lower().endswith(ext) for ext in image_extensions):
                    img_path = os.path.join(subfolder_path, img_name)
                    print(f"Reading image: {img_path}")
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    image = cv2.resize(image, (64, 64))
                    data.append(image)
                    labels.append(i)

# Converting data and labels to numpy arrays
data = np.array(data, dtype='float32')
data = data.reshape(data.shape[0], 64, 64, 1)
labels = np.array(labels)

data /= 255.0

labels = to_categorical(labels, num_classes=len(gestures))

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f'Training data shape: {X_train.shape}')
print(f'Testing data shape: {X_test.shape}')


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(gestures), activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

model.save('hand_gesture_model.keras')
print("Model saved as 'hand_gesture_model.keras'")