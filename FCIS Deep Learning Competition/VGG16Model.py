# VGG 16 ele gab 90%
# Maiiiii
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle
import os
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as np_utils
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16, Xception, VGG19
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
from PIL import Image
import os
# import urllib2
from PIL import Image
from cv2 import resize

# public
# region
submitFile = pd.read_csv('/content/drive/My Drive/submit.csv')
TRAIN_DIR = '/content/drive/My Drive/return 0/neural data/train'
TEST_DIR = '/content/drive/My Drive/return 0/neural data/test'
weight_Path = '/content/drive/My Drive/vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMG_SIZE = 224


# endregion


def create_train_data():
    training_data = []
    sortedTrain = []
    label = 1
    for folder in os.listdir(TRAIN_DIR):
        sortedTrain.append(folder)

    sortedTrain.sort()
    for folder in tqdm(sortedTrain):
        print(folder)
        folder_path = os.path.join(TRAIN_DIR, folder)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            img_data = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_data = image.img_to_array(img_data)
            img_data = preprocess_input(img_data)
            training_data.append(np.array([img_data, label]))  # el img el resized  - label
        label = label + 1
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, img)

        img_data = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_data = image.img_to_array(img_data)
        # img_data=img_data.reshape((1,img_data[0],img_data[1],img_data[2]))
        img_data = preprocess_input(img_data)
        testing_data.append(img_data)
    np.save('test_data.npy', testing_data)
    return testing_data


if (os.path.exists('train_data.npy')):
    train_data = np.load('train_data.npy', allow_pickle=True)
    print("train data loaded")
else:
    train_data = create_train_data()
    print("train data created")

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
    print("test data loaded")
else:
    test_data = create_test_data()
    print("test data created")

X_train = np.array([i[0] for i in train_data])
y_train = [i[1] - 1 for i in train_data]

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=0,
                                                      shuffle=True)

Y_train = np_utils.to_categorical(y_train)
Y_valid = np_utils.to_categorical(y_valid)

model = VGG16(include_top=False, weights=weight_Path, input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=1365)
model.summary()

# transfer_layer = model.get_layer('block5_pool')
transfer_layer = model.layers[-1]
conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
new_model = Sequential()
new_model.add(conv_model)

new_model.add(Flatten())
new_model.add(Dense(1024, activation='relu'))
# new_model.add(Dropout(0.2))
new_model.add(Dense(512, activation='relu'))
new_model.add(Dropout(0.2))
new_model.add(Dense(10, activation='softmax'))

for layer in conv_model.layers:
    trainable = ('block5' in layer.name)  # or 'block4' in layer.name)
    layer.trainable = trainable

for layer in conv_model.layers:
    print("{0}:\t{1}".format(layer.trainable, layer.name))

# for layer in conv_model.layers:
#     layer.trainable = False


new_model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# accMatrix = new_model.fit(X_train, Y_train,batch_size=64, epochs=30, verbose=2, validation_data=(X_valid, Y_valid))


dataAugmentaion = ImageDataGenerator(rotation_range=30, zoom_range=0.20,
                                     fill_mode="nearest", shear_range=0.20, horizontal_flip=True,
                                     width_shift_range=0.1, height_shift_range=0.1)

new_model.fit_generator(dataAugmentaion.flow(X_train, Y_train, batch_size=64), validation_data=(X_valid, Y_valid),
                        steps_per_epoch=len(X_train) / 64, epochs=30)

new_model.save('vggModel.h5')
print('model saved')

# if (os.path.exists('vggModel.h5')):
#   new_model=load_model('vggModel.h5')
#   print('model loaded')

print('predicting...')
prediction = new_model.predict(test_data)
predicted_label = np.argmax(prediction, axis=1)
predicted_label = predicted_label + 1

testId = []
for img in os.listdir(TEST_DIR):
    testId.append(img)

submitFile['Id'] = testId
submitFile['Label'] = predicted_label
submitFile.to_csv('submit.csv', index=False)