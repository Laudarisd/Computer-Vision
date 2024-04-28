#-*- coding: utf-8 -*-
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input?version=nightly#raises
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input?hl=ko&version=nightly#returns
# resnet preprocess_input 이랑 mobilenet preprocess_input return 값이 다르기 때문에 preprocess_input을 어떻게 적용해야함??
import tensorflow as tf
import glob
import cv2
import numpy as np
# from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
# from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

from keras.backend.tensorflow_backend import set_session

def set_gpu_option(which_gpu, fraction_memory):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
    config.gpu_options.visible_device_list = which_gpu
    set_session(tf.Session(config=config))

set_gpu_option("0", 0.8)

def ensemble(models_list, model_input):
    outputs = [model(model_input) for model in models]

    yAvg = tf.keras.layers.average(outputs)

    model = tf.keras.Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return model

models = []

model1_path = '/data/backup/pervinco_2020/model/cu50/2020.05.06_09:17_keras/resnet50.h5'
model2_path = '/data/backup/pervinco_2020/model/cu50/2020.05.06_09:17_keras/mobilenet.h5'

model1 = tf.keras.models.load_model(model1_path)
model2 = tf.keras.models.load_model(model2_path)

models.append(model1)
models.append(model2)

IMAGE_RESIZE = 224
CHANNELS = 3
model_input = tf.keras.Input(shape=(IMAGE_RESIZE, IMAGE_RESIZE, CHANNELS))
model = ensemble(models, model_input)

# load image
img_path = sorted(glob.glob('/data/backup/pervinco_2020/test_code/test_img/*.jpg'))

for i in img_path:
    file_name = i.split('/')[-1]
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)

    predictions = model.predict(img)
    # print(predictions)
    predictions = np.array(predictions)
    print(predictions)