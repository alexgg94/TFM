import os
import numpy as np
import keras
from PIL import Image
import cv2
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.utils import np_utils
from keras.models import model_from_json

"""
Resize the given image to 50x50
"""
def ImageResizer(image):
    return cv2.resize(image,(50,50))

"""
Generates new images by flipping (horizontally, vertically and both at the same time) the original one
"""
def ImageGenerator(image, folder_name, image_name):
    cv2.imwrite("data/processed/"+ folder_name +"/" + image_name, image)
    cv2.imwrite("data/processed/"+ folder_name +"/h-" + image_name, cv2.flip(image, 0))
    cv2.imwrite("data/processed/"+ folder_name +"/v-" + image_name, cv2.flip(image, 1))
    cv2.imwrite("data/processed/"+ folder_name +"/hv-" + image_name, cv2.flip(image, -1))


if __name__ == "__main__":
    for folder in os.listdir("data/raw"):
        for image_name in os.listdir("data/raw/"+folder):
            try:
                print("Processing " + image_name)
                image = cv2.imread("data/raw/"+ folder +"/" + image_name)
                resized_image = ImageResizer(image)
                if not os.path.exists("data/processed/"+ folder +"/"):
                    os.makedirs("data/processed/"+ folder +"/")
                ImageGenerator(resized_image, folder, image_name)
            except:
                print("Could not process " + image_name)

    data = []
    labels = []

    for cow in os.listdir("data/processed/Cow"):
        try:
            image = cv2.imread("data/processed/Cow/" + cow)
            data.append(np.array(image))
            labels.append(0)
        except:
            print("Could not process " + cow)

    for no_cow in os.listdir("data/processed/NoCow"):
        try:
            image = cv2.imread("data/processed/NoCow/" + no_cow)
            data.append(np.array(image))
            labels.append(1)
        except:
            print("Could not process " + no_cow)

    data = np.array(data)
    labels = np.array(labels)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    y = keras.utils.np_utils.to_categorical(labels, len(np.unique(labels)))
    x = data.astype('float32')/255
    model.fit(x, y, batch_size=128, epochs=40, verbose=1)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
