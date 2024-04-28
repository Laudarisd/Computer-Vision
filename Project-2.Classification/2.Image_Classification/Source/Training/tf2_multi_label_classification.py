import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

csv_file = '/data/backup/pervinco_2020/datasets/miml_dataset/miml_labels_2.csv'
training_set = pd.read_csv(csv_file)
training_set["labels"] = training_set["labels"].apply(lambda x:x.split(","))

img_dir = '/data/backup/pervinco_2020/datasets/miml_dataset/images'
classes=['desert', 'sunset', 'trees', 'mountains', 'sea']

print("total")
print(training_set)

# print(training_set["Filenames"][0])

def basic_processing(dataset, img_dir, classes):
    images = []
    labels = []
    classes = dict((name, index) for index, name in enumerate(classes))

    print(classes)

    for i in range(0, len(dataset)):
        image = dataset["Filenames"][i]
        image = img_dir + '/' + image

        label = dataset["labels"][i]
        one_hot_label = np.zeros(len(classes))

        if len(label) != 1:
            for l in range(0, len(label)):
                one_hot_label[classes[label[l]]] = 1
            label = one_hot_label

        else:
            one_hot_label[classes[label[0]]] = 1
            label = one_hot_label

        images.append(image)
        labels.append(label)

    print("Images sample: ", images[:10])
    print("Labels sample: ", labels[:10])
    print("Total Images, Labels : ", len(images), len(labels))
    
    return images, labels

images, labels = basic_processing(training_set, img_dir, classes)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = keras.applications.xception.preprocess_input(image)

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    
    return preprocess_image(image)

def make_tf_dataset(images, labels):
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = image_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    lable_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))

    image_label_ds = tf.data.Dataset.zip((image_ds, lable_ds))

    return image_label_ds

ds = make_tf_dataset(images, labels)
ds = ds.shuffle(buffer_size=1000)
ds = ds.repeat().batch(32)

# for i in ds.take(1):
#     print(i)

# data_generator = ImageDataGenerator(preprocessing_function=keras.applications.xception.preprocess_input)
# train_generator = data_generator.flow_from_dataframe(dataframe = training_set,
#                                                     directory=img_dir,
#                                                     x_col="Filenames",
#                                                     y_col="labels",
#                                                     class_mode="categorical",
#                                                     classes=['desert', 'sunset', 'trees', 'mountains', 'sea'],
#                                                     target_size=(224,224),
#                                                     batch_size=3)

# print(next(train_generator))

base_model = keras.applications.xception.Xception(input_shape=(224, 224, 3),
                                                  weights="imagenet",
                                                  include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(len(classes), activation="sigmoid")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(ds,
                    epochs=10,
                    shuffle=False,
                    steps_per_epoch=100)

model.save('/data/backup/pervinco_2020/model/test_model/mlc_test_model2.h5')