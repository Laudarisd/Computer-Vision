import tensorflow as tf
import pathlib
import random
import os
import datetime
import json
import time


def basic_processing(ds_path, is_training):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]
    len_images = len(images)

    if is_training:
        random.shuffle(images)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    labels_len = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=labels_len, dtype='float32')

    return images, labels, len_images, labels_len


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.efficientnet.preprocess_input(image)

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


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


# def build_model():
#     base_model = tf.keras.applications.EfficientNetB5(input_shape=(IMG_SIZE, IMG_SIZE, 3),
#                                 weights="imagenet", # noisy-student
#                                 include_top=False)
#     avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
#     output = tf.keras.layers.Dense(train_labels_len, activation="softmax")(avg)
#     model = tf.keras.Model(inputs=base_model.input, outputs=output)

#     for layer in base_model.layers:
#         layer.trainable = True

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     return model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


model_name = "EfficientNet-B5"
dataset_name = 'beverage'
train_dataset_path = '/data/Auged_datasets/' + dataset_name + '/train_3'
valid_dataset_path = '/data/Auged_datasets/' + dataset_name + '/valid_3'

train_images, train_labels, train_images_len, train_labels_len = basic_processing(train_dataset_path, True)
valid_images, valid_labels, valid_images_len, valid_labels_len = basic_processing(valid_dataset_path, False)

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
IMG_SIZE = 224
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 3
TRAIN_STEP_PER_EPOCH = int(train_images_len / BATCH_SIZE)
VALID_STEP_PER_EPOCH = int(valid_images_len / BATCH_SIZE)

saved_path = './model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_tf2'
weight_file_name = '{epoch:02d}-{val_accuracy:.2f}.hdf5'

if not(os.path.isdir(saved_path + dataset_name + '/' + time)):
    os.makedirs(os.path.join(saved_path + dataset_name + '/' + time))

    f = open(saved_path + dataset_name + '/' + time + '/README.txt', 'w')
    f.write('IMG_SIZE = ' + str(IMG_SIZE) + '\n')
    f.write(train_dataset_path + '\n')
    f.write(valid_dataset_path + '\n')
    f.write("Model : " + model_name)
    f.close()

else:
    pass

train_ds = make_tf_dataset(train_images, train_labels)
valid_ds = make_tf_dataset(valid_images, valid_labels)

train_ds = train_ds.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
valid_ds = valid_ds.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)

with strategy.scope():
    base_model = tf.keras.applications.EfficientNetB5(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                      weights="imagenet", # noisy-student
                                                      include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(train_labels_len, activation="softmax")(avg)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
checkpoint_path = saved_path + dataset_name + '/' + time + '/' + weight_file_name
cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor='val_accuracy',
                                                    save_best_only=True,
                                                    mode='max')
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)    

history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                    shuffle=False,
                    validation_data=valid_ds,
                    validation_steps=VALID_STEP_PER_EPOCH,
                    verbose=1,
                    workers=1,
                    callbacks=[cb_early_stopper, cb_checkpointer, lr_schedule])
                    # callbacks=[cb_early_stopper, lr_schedule])

model.save(saved_path + dataset_name + '/' + time + '/' + dataset_name + '.h5')
