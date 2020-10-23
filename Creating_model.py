import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import cv2
import warnings
from PIL import Image
import os
from keras.preprocessing import image


warnings.filterwarnings(action='once')
gestures = {
    "Peace": 0,
    "Palm": 1,
    "Fist": 2,
    "Thumbs-up": 3,
    "L": 4
}
gestures_map = {
    "Pe": "Peace",
    "Pa": "Palm",
    "Fi": "Fist",
    "Th": "Thumbs-up",
    "L_": "L"
}

train_path = "C:/Users/anani/Desktop/DPNS/train_set/"
val_path = "C:/Users/anani/Desktop/DPNS/val_set/"
test_path = "C:/Users/anani/Desktop/DPNS/test_set/"


def process_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img


def process_data(x_data, y_data):
    x_data = np.array(x_data, dtype='float32')
    x_data = np.stack((x_data,) * 3, axis=-1)
    x_data /= 255
    y_data = np.array(y_data)
    y_data = keras.utils.to_categorical(y_data)

    return x_data, y_data


def get_data(folder_path):
    x_data = []
    y_data = []

    for directory, subdirectory, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(directory, file)
            gesture_name = gestures_map[file[:2]]
            y_data.append(gestures[gesture_name])
            x_data.append(process_image(image_path))

    x_data, y_data = process_data(x_data, y_data)
    return x_data, y_data


train_x, train_y = get_data(train_path)
val_x, val_y = get_data(val_path)
test_x, test_y = get_data(test_path)

# kernel_size = (5, 5) - goleminata na filterot
model = Sequential()
model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25, seed=21))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, batch_size=16, validation_data=(val_x, val_y), verbose=1)

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(test_acc)

model.save("gesture_model.h5")

