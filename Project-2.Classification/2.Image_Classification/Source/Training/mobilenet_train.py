# -*- coding: utf-8 -*-
import tensorflow as tf
import pathlib
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

print("Tensorflow version" + tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dir = '/home/barcelona/pervinco/datasets/100-bird-species/train'
valid_dir = '/home/barcelona/pervinco/datasets/100-bird-species/valid'

train_data_dir = pathlib.Path(train_dir)
# print(train_data_dir)
valid_data_dir = pathlib.Path(valid_dir)
# print(valid_data_dir)


train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
print(train_image_count)
valid_image_count = len(list(valid_data_dir.glob('*/*.jpg')))
print(valid_data_dir)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
print(CLASS_NAMES)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)

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
model.add(MobileNet(include_top=False, pooling='avg', weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(Dense(len(CLASS_NAMES), activation='softmax'))
model.layers[0].trainable = True
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit_generator(train_data_gen,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_data=valid_data_gen,
                              validation_steps=STEPS_PER_EPOCH,
                              epochs=10)

model.save('/home/barcelona/ImageClassification/model/saved_model/Mobilenet.h5')
