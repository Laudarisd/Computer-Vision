import tensorflow as tf
import glob
import os
import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session

def set_gpu_option(which_gpu, fraction_memory):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
    config.gpu_options.visible_device_list = which_gpu
    set_session(tf.Session(config=config))
    return


set_gpu_option("0", 0.8)

DATASET_NAME = 'total_split'
MODEL_NAME = DATASET_NAME
model_name = "ResNet50"

train_dir = '/data/backup/pervinco_2020/Auged_datasets/' + DATASET_NAME + '/train_2'
valid_dir = '/data/backup/pervinco_2020/Auged_datasets/' + DATASET_NAME + '/valid_2'
NUM_CLASSES = len(glob.glob(train_dir + '/*'))

CHANNELS = 3
IMAGE_RESIZE = 224
NUM_EPOCHS = 30
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 3

saved_path = '/data/backup/pervinco_2020/model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_keras'
weight_file_name = '{epoch:02d}-{val_acc:.2f}.hdf5'

if not (os.path.isdir(saved_path + DATASET_NAME + '/' + time)):
    os.makedirs(os.path.join(saved_path + DATASET_NAME + '/' + time))
else:
    pass

resnet_weights_path = '/data/backup/pervinco_2020/source/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = tf.keras.models.Sequential()
model.add(tf.keras.applications.ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
model.layers[0].trainable = True
model.summary()

optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(train_dir,
                                                     target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

valid_generator = data_generator.flow_from_directory(valid_dir,
                                                     target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

# print(train_generator[0])


cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=saved_path + DATASET_NAME + '/' +
                                                              time + '/' + weight_file_name,
                                                     monitor='val_acc', save_best_only=True, mode='auto')

fit_history = model.fit(train_generator,
                        steps_per_epoch=train_generator.n / BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        shuffle=False,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.n / BATCH_SIZE,
                        callbacks=[cb_early_stopper, cb_checkpointer])

model.save(saved_path + DATASET_NAME + '/' + time + '/' + MODEL_NAME + '.h5')

f = open(saved_path + DATASET_NAME + '/' + time + '/README.txt', 'w')
f.write(train_dir + '\n')
f.write(valid_dir + '\n')
f.write("Model : " + model_name)
