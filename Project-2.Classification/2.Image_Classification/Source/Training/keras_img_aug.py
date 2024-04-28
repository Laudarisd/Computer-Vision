import tensorflow as tf
import numpy as np
import glob
import os

DATASET_NAME = 'cu50'
path = sorted(glob.glob('/home/barcelona/pervinco/datasets/'+ DATASET_NAME + '/added/*/*'))
output_path = '/home/barcelona/pervinco/datasets/' + DATASET_NAME + '/aug/'
print('label num : ', len(path))

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360,
                                                                 width_shift_range=0.2,
                                                                 height_shift_range=0.2,
                                                                 zoom_range=0.1)


# path = sorted(glob.glob('/home/barcelona/pervinco/datasets/four_shapes/predict/*/*.png'))
# print(len(path))

for image in path:
    folder = image.split('/')[-2]
    print('processing', folder, image)
    image = tf.keras.preprocessing.image.load_img(image)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image, (224, 224))
    image = np.expand_dims(image, 0)
    data_generator.fit(image)

    if not (os.path.isdir(output_path + folder)):
        os.makedirs(output_path + folder)

    for x, val in zip(data_generator.flow(image, save_to_dir=output_path + folder, save_prefix=folder, save_format='jpg'), range(60)):
        pass