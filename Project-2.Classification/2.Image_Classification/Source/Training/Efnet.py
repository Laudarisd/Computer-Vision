import cv2
import numpy as np
import tensorflow as tf
import pathlib
import keras
from tensorflow.keras import backend as K
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from math import ceil
from efficientnet.keras import EfficientNetB0, preprocess_input

print("Tensorflow version : " + tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data_dir = '/home/barcelona/pervinco/datasets/cu50/train5'
valid_data_dir = '/home/barcelona/pervinco/datasets/cu50/valid5'

train_data_dir = pathlib.Path(train_data_dir)
# print(train_data_dir)
valid_data_dir = pathlib.Path(valid_data_dir)
# print(valid_data_dir)


train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
print(train_image_count)
valid_image_count = len(list(valid_data_dir.glob('*/*.jpg')))
print(valid_image_count)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
print(CLASS_NAMES)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
TRAIN_STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)
VALID_STEPS_PER_EPOCH = np.ceil(valid_image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     # shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

valid_data_gen = image_generator.flow_from_directory(directory=str(valid_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     # shuffle=False,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

model = Sequential()
model.add(EfficientNetB0(include_top=False, pooling='avg', weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(Dense(len(CLASS_NAMES), activation='softmax'))
model.layers[0].trainable = True
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit_generator(train_data_gen,
                              steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                              validation_data=valid_data_gen,
                              validation_steps=VALID_STEPS_PER_EPOCH,
                              epochs=10)

model.save('/home/barcelona/ImageClassification/model/saved_model/EfficientNet_cu50.h5')