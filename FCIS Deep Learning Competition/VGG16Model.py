
#VGG 16 ele gab 90%
#Maiiiii
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from random import shuffle
import os
import cv2
from tqdm import tqdm
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import tensorflow.keras.utils as np_utils
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
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
TEST_DIR = '/content/drive/My Drive/return 0/labeled test new'
TEST_DIR2 = '/content/drive/My Drive/return 0/neural data/test'
weight_Path = '/content/drive/My Drive/vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMG_SIZE = 224


# endregion


def create_train_data():
    training_data = []
    sortedTrain=[]
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
            img_data = preprocess_input(img_data)  # by7wlha gbr w y3mlha scale
            training_data.append(np.array([img_data, label]))  # el img el resized-label
        label = label + 1
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data



def create_test_data():
    testing_data = []
    sortedTest=[]
    label = 1
    for folder in os.listdir(TEST_DIR):  
        sortedTest.append(folder)

    sortedTest.sort()  
    for folder in tqdm(sortedTest):
        print(folder)
        folder_path = os.path.join(TEST_DIR, folder)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            img_data = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_data = image.img_to_array(img_data)
            img_data = preprocess_input(img_data)  # by7wlha gbr w y3mlha scale
            testing_data.append(np.array([img_data, label]))  # el img el resized-label
        label = label + 1
    np.save('test_data.npy',testing_data)

    return testing_data


if (os.path.exists('train_data.npy')):
    train_data = np.load('train_data.npy', allow_pickle=True)
    print ("train data loaded")
else:
    train_data = create_train_data()
    print("train data created")

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
    print ("test data loaded")
else:
    test_data = create_test_data()
    print("test data created")


X_train = np.array([i[0] for i in train_data])
y_train = [i[1] - 1 for i in train_data]

X_test = np.array([i[0] for i in test_data])
y_test = [i[1] - 1 for i in test_data]

#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=0, shuffle=True)


Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
Y_valid = np_utils.to_categorical(y_valid)

model = VGG16(include_top=False, weights=weight_Path,input_shape=(IMG_SIZE,IMG_SIZE,3), classes=1365)
# model=Xception(include_top=True,weights='imagenet')
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
    trainable = ('block5' in layer.name )#or 'block4' in layer.name) 
    layer.trainable = trainable

# for layer in conv_model.layers:
#     layer.trainable = False
for layer in conv_model.layers:
    print("{0}:\t{1}".format(layer.trainable, layer.name))
# conv_model.trainable = False

# for layer in conv_model.layers:
#     layer.trainable = False

# for layer in conv_model.layers:
#     print("{0}:\t{1}".format(layer.trainable, layer.name))


optimizer = Adam(lr=1e-5)
# optimizer_fine = Adam(lr=1e-7)
optimizer_fine = Adam(lr=1e-4)
print(optimizer_fine)
#new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)
new_model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# accMatrix = new_model.fit(X_train, Y_train,batch_size=64, epochs=30, verbose=2, validation_data=(X_valid, Y_valid))


dataAugmentaion = ImageDataGenerator(rotation_range = 30, zoom_range = 0.20, 
fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True, 
width_shift_range = 0.1, height_shift_range = 0.1)

new_model.fit_generator(dataAugmentaion.flow(X_train, Y_train,batch_size = 64), validation_data = (X_valid, Y_valid),
steps_per_epoch = len(X_train) / 64, epochs = 30)



# # loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
# # print("Test Loss", loss_and_metrics[0])
# # print("Test Accuracy", loss_and_metrics[1] * 100, "%")

new_model.save('MEXceptionModel.h5')
print('model saved')

# if (os.path.exists('MEvgg16Model.h5')):
#   new_model=load_model('MEvgg16Model.h5')
#   print('model loaded')
print('predicting...')

print(X_test.shape)
# prediction = new_model.predict(test_data)
prediction = new_model.predict(X_test)
predicted_label=np.argmax(prediction,axis=1)
predicted_label = predicted_label + 1

testId=[]
# for img in os.listdir(TEST_DIR2): #kda 8alat
#   testId.append(img)

sortedTest=[]
for folder in os.listdir(TEST_DIR):
    sortedTest.append(folder)
sortedTest.sort()  
for folder in sortedTest:
    folder_path = os.path.join(TEST_DIR, folder)
    for img in os.listdir(folder_path):
        testId.append(img)

submitFile['Id']=testId    
print('predicted_label.shape',predicted_label.shape)
submitFile['Label'] = predicted_label
submitFile.to_csv('submit.csv', index=False)

correct = 0
for i in range(len(Y_test)):
    prediction[i, :] = np.where(prediction[i, :] == max(prediction[i, :]), 1, 0)
    if np.array_equal(prediction[i], Y_test[i]):
        correct += 1

print("Overall Accuracy", (correct / len(Y_test)) * 100, "%")