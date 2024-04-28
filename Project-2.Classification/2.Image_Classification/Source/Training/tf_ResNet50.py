# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import cv2
from PIL import Image
import pathlib
import glob
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense

DATASET_NAME = 'dog_cls'
IMG_TYPE = '.jpg'
BATCH_SIZE = 32

IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 10
EARLY_STOP_PATIENCE = 5
saved_path = '/home/barcelona/pervinco/model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_tf'
model_name = DATASET_NAME
weight_file_name = '{epoch:02d}-{val_acc:.2f}.hdf5'


def make_np(data_list, class_label):
    labels = []
    images = []
    for img in data_list:
        label = img.split('/')[-2]

        if label in class_label:
            labels.append(class_label.index(label))

        img = Image.open(img)
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img = np.array(img)
        images.append(img)

    np_images = np.array(images)
    np_labels = np.array(labels)
    np_labels = np_labels.reshape(len(labels), 1)

    return np_images, np_labels


def data_to_np(train_dir, valid_dir):
    class_label = []

    train_data_list = glob.glob(str(train_dir) + '/*/*' + IMG_TYPE)
    valid_data_list = glob.glob(str(valid_dir) + '/*/*' + IMG_TYPE)
    train_label_list = glob.glob(str(train_dir) + '/*')

    for i in range(0, len(train_label_list)):
        class_label.append(train_label_list[i].split('/')[-1])

    class_label = sorted(class_label)
    print('Dataset Label : ', class_label)

    train_images, train_labels = make_np(train_data_list, class_label)
    valid_images, valid_labels = make_np(valid_data_list, class_label)

    return (train_images, train_labels), (valid_images, valid_labels), len(train_label_list)


if __name__ == '__main__':
    if not(os.path.isdir(saved_path + DATASET_NAME + '/' + time)):
        os.makedirs(os.path.join(saved_path + DATASET_NAME + '/' + time))
    else:
        pass

    train_dir = pathlib.Path('/home/barcelona/pervinco/datasets/' + DATASET_NAME + '/train')
    total_train_data = len(list(train_dir.glob('*/*' + IMG_TYPE)))
    print('total train data : ', total_train_data)

    valid_dir = pathlib.Path('/home/barcelona/pervinco/datasets/' + DATASET_NAME + '/valid')
    total_valid_data = len(list(valid_dir.glob('*/*' + IMG_TYPE)))
    print('total validation data : ', total_valid_data)

    (x_train, y_train), (x_test, y_test), NUM_CLASSES = data_to_np(train_dir, valid_dir)

    print('train images, labels', x_train.shape, y_train.shape)
    print('validation images, labels', x_test.shape, y_test.shape)

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    SHUFFLE_BUFFER_SIZE = 1000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    '''
    이 커널은 top layer를 제외한 ResNet50에 사전 훈련 된 가중치를 사용하여 transfer learning을 합니다.
    '''
    resnet_weights_path = '/home/barcelona/pervinco/source/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.layers[0].trainable = True
    model.summary()
    
    optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
    cb_checkpointer = ModelCheckpoint(filepath=saved_path + DATASET_NAME + '/' + time + '/' + weight_file_name,
                                      monitor='val_acc', save_best_only=True, mode='auto')

    model.fit_generator(data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=total_train_data / BATCH_SIZE,
                        validation_data=data_generator.flow(x_test, y_test, batch_size=BATCH_SIZE),
                        validation_steps=total_valid_data / BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[cb_early_stopper, cb_checkpointer])

    model.save(saved_path + DATASET_NAME + '/' + time + '/' + model_name + '.h5')

