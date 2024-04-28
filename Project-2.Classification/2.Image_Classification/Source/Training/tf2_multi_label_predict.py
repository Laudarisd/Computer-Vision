# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import os
import glob
import datetime
import numpy as np
import cv2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print("True")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# classes = ['111', '112', '113', '114', '115', 
#            '221', '222', '223', '224', '225']
classes=['desert', 'sunset', 'trees', 'mountains', 'sea']
print(classes)

model = Sequential()
model.add(InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet'))
# model.add(Dense(10, activation='sigmoid'))
model.add(Dense(len(classes), activation='sigmoid'))
model.layers[0].trainable = True
model.summary()

optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = optimizers.Adam()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# model.load_weights('/data/backup/pervinco_2020/model/multi_label/2020.05.07_17:27_keras/08-1.00.hdf5')
model.load_weights('/data/backup/pervinco_2020/model/multi_label_cls/2020.05.11_10:33_keras/10-1.00.hdf5')

# model = tf.keras.models.load_model('/data/backup/pervinco_2020/model/multi_label/test.h5')

test_imgs = sorted(glob.glob('/data/backup/pervinco_2020/datasets/multi_label_cls/test_imgs/*.jpg'))
for test_img in test_imgs:
    file_name = test_img.split('/')[-1]
    test_img = cv2.imread(test_img)
    test_img = cv2.resize(test_img, (224, 224))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = preprocess_input(test_img)
    test_img = np.expand_dims(test_img, axis=0)

    prediction = model.predict(test_img, steps=1)

    # prediction = model.predict(test_img, steps=1)
    # index = np.argmax(prediction[0])
    # score = prediction[0][index]
    # result = classes[np.argmax(prediction[0])]
    # print(result, score)

    top_3 = np.argsort(prediction[0])[:-4:-1]
    print("==========================" + file_name + "==========================")
    for i in range(3):
        print("{}".format(classes[top_3[i]])+" ({:.3})".format(prediction[0][top_3[i]]))