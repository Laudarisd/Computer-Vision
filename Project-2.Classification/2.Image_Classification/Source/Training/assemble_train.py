#-*- coding: utf-8 -*-
# load weight은 model을 선언한 다음 가중치를 넣어줄 수 있게끔 사용
# load model은 model + weight가 저장된 형태 그대로 불러서 사용
# https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko#%EB%AA%A8%EB%8D%B8_%EC%A0%84%EC%B2%B4%EB%A5%BC_%EC%A0%80%EC%9E%A5%ED%95%98%EA%B8%B0

# Reference
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit_generator
# https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb
# https://medium.com/randomai/ensemble-and-store-models-in-keras-2-x-b881a6d7693f
import os
import cv2
import glob
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.python.keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

'''
h_parameters
'''
IMAGE_RESIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
NUM_EPOCHS = 3
EARLY_STOP_PATIENCE = 3

'''
saving model options
'''
saved_path = '/data/backup/pervinco_2020/model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_keras'
weight_file_name = '{epoch:02d}-{val_acc:.2f}.hdf5'
dataset_name = 'cu50'
train_dir = '/data/backup/pervinco_2020/datasets/' + dataset_name + '/train5'
valid_dir = '/data/backup/pervinco_2020/datasets/' + dataset_name + '/valid5'
NUM_CLASSES = len(glob.glob(train_dir + '/*'))

'''
input_tensor shape
'''
model_input = tf.keras.Input(shape=(IMAGE_RESIZE, IMAGE_RESIZE, CHANNELS))

def set_gpu_option(which_gpu, fraction_memory):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
    config.gpu_options.visible_device_list = which_gpu
    set_session(tf.Session(config=config))

set_gpu_option("0", 0.8)

def build_resnet50(model_input):
    x = ResNet50(include_top=False, pooling='avg', weights='imagenet')(model_input)
    x = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(model_input, x, name="resnet50")

    model.layers[0].trainable = True

    return model


def build_mobilenet(model_input):
    x = MobileNet(include_top=False, pooling='avg', weights='imagenet')(model_input)
    x = Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(model_input, x, name="mobilenet")

    model.layers[0].trainable = True

    return model

def compile_and_train(model, model_name):
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
    cb_checkpointer = ModelCheckpoint(filepath=saved_path + dataset_name + '/' + time +
                                      '/' +  model_name + '_' + weight_file_name,
                                      monitor='val_acc', save_best_only=True, mode='auto')

    
    if not(os.path.isdir(saved_path + dataset_name + '/' + time)):
        os.makedirs(os.path.join(saved_path + dataset_name + '/' + time))
    else:
        pass

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n / BATCH_SIZE,
                                  epochs=NUM_EPOCHS,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n / BATCH_SIZE,
                                  callbacks=[cb_early_stopper, cb_checkpointer])

    model.save(saved_path + dataset_name + '/' + time + '/' + model_name + '.h5')    


if __name__ == "__main__":

    data_generator = ImageDataGenerator(preprocessing_function=resnet_preprocess)
    train_generator = data_generator.flow_from_directory(train_dir,
                                                        target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')
    validation_generator = data_generator.flow_from_directory(valid_dir,
                                                            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical')
    resnet_model = build_resnet50(model_input)
    compile_and_train(resnet_model, 'resnet50')


    data_generator = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
    train_generator = data_generator.flow_from_directory(train_dir,
                                                        target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')
    validation_generator = data_generator.flow_from_directory(valid_dir,
                                                            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical')
    mobilenet_model = build_mobilenet(model_input)
    compile_and_train(mobilenet_model, 'mobilenet')