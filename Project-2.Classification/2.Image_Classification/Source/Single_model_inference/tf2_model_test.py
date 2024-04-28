import tensorflow as tf
import glob
from tensorflow import keras
import numpy as np
import cv2
import os
import pandas as pd
from efficientnet.tfkeras import preprocess_input
# from tensorflow.keras.applications.resnet50 import preprocess_input

model_path = '/data/data/interminds/modeldatasets/2020.08.31_05:10_tf2/datasets.h5'

#dataset_name = model_path.split('/')[-3]
dataset_name = 'test'
test_img_path = '/data/data/interminds/' + dataset_name + '/img/*.jpg'
#class_path = '/home/sudip/mini_project/img_classification/root/' + dataset_name
#class_path = '/home/sudip/mini_project/img_classification/root/'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
#   except RuntimeError as e:
#     print(e)

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
#    tf.config.experimental.set_memory_growth(gpus[0], True)
#  except RuntimeError as e:
#    print(e)

model = tf.keras.models.load_model(model_path)
model.summary()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
  except RuntimeError as e:
      print(e)






#CLASS_NAMES = sorted(os.listdir(class_path))
#print(CLASS_NAMES)
#print(len(CLASS_NAMES))



df = pd.read_csv('/data/data/interminds/labels.txt', sep = ' ', index_col=False, header=None)
CLASS_NAMES = df[0].tolist()
CLASS_NAMES = sorted(CLASS_NAMES)

print(len(CLASS_NAMES))

test_imgs = sorted(glob.glob(test_img_path))
print(len(test_imgs))

for img in test_imgs:
    file_name = img.split('/')[-1]
    image = cv2.imread(img)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    

    predictions = model.predict(image, steps=1)
    index = np.argmax(predictions[0])
    name = str(CLASS_NAMES[index])
    score = str(predictions[0][index])

    print(file_name, name, score)

    
