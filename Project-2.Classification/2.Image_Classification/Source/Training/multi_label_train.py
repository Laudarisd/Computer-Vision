# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import os
import datetime
import numpy as np
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

dataset_name = 'multi_label_cls'
BATCH_SIZE = 32
NUM_EPOCHS = 10
IMAGE_SIZE = 224
EARLY_STOP_PATIENCE = 3
saved_path = '/data/backup/pervinco_2020/model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_keras'
weight_file_name = '{epoch:02d}-{accuracy:.2f}.hdf5'

if not(os.path.isdir(saved_path + dataset_name + '/' + time)):
    os.makedirs(os.path.join(saved_path + dataset_name + '/' + time))
else:
    pass

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print("True")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        

training_set = pd.read_csv('/data/backup/pervinco_2020/datasets/custom_miml.csv')
training_set["labels"] = training_set["labels"].apply(lambda x: x.split(","))

print(training_set.head())

img_dir = "/data/backup/pervinco_2020/datasets/multi_label_cls/images"

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = data_generator.flow_from_dataframe(dataframe = training_set,
                                                    directory=img_dir,
                                                    x_col="Filenames",
                                                    y_col="labels",
                                                    class_mode="categorical",
                                                    classes=['1850', '3211', '3715', '5203', '5601', '8584'],
                                                    target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                    batch_size=32)

cb_early_stopper = EarlyStopping(monitor='loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath=saved_path + dataset_name + '/' + time + '/' + weight_file_name,
                                  monitor='accuracy', save_best_only=True, mode='auto')

model = Sequential()
model.add(InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(6, activation='sigmoid'))
model.layers[0].trainable = True
model.summary()

optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = optimizers.Adam()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

fit_history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n / BATCH_SIZE,
                                  epochs=NUM_EPOCHS,
                                #   validation_data=validation_generator,
                                #   validation_steps=validation_generator.n / BATCH_SIZE,
                                  callbacks=[cb_early_stopper, cb_checkpointer])


model.save(saved_path + dataset_name + '/' + time + '/' + dataset_name + '.h5')