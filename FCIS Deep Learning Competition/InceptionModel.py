# Nourhan with data augmentation

import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from random import shuffle
import os
import cv2
import numpy as np
from tqdm import tqdm  # Or from tqdm import tqdm if not jupyter notebook
from sklearn.model_selection import train_test_split
from PIL import Image
import tflearn
from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras import optimizers
import pandas as pd


# public
# region

TRAIN_DIR = '/content/drive/My Drive/return 0/neural data/train'
TEST_DIR = '/content/drive/My Drive/return 0/neural data/test'
submitFile= pd.read_csv('/content/drive/My Drive/return 0/neural data/submit.csv',keep_default_na=True)
labeledTest = '/content/drive/My Drive/return 0/labeled test new'

IMG_SIZE = 299
LR = 0.001
MODEL_NAME = 'identify-places-cnn'
def create_train_data():#lesaa el label 3ayz yt3ml 3aleh 1 hot encoding***
    training_data = []
    sortedFolders = []
    label = 1
    for folder in tqdm(os.listdir(TRAIN_DIR)):  # folder name 10 classes
        sortedFolders.append(folder)
    sortedFolders.sort()
    for folder in sortedFolders:
        folder_path = os.path.join(TRAIN_DIR, folder)
        for img in tqdm(os.listdir(folder_path)):# each folder has set of images
            img_path = os.path.join(folder_path, img)
            img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_data = img_data[:, :, [2, 1, 0]]
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            # img = image.load_img(img_path, target_size=(299, 299))
            # x = image.img_to_array(img)
            # x = preprocess_input(x)
            training_data.append([np.array(img_data), label])  # el img el resized-label
        label = label + 1
    shuffle(training_data)
    np.save('train_data_299.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = img_data[:, :, [2, 1, 0]]
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        # img = image.load_img(path, target_size=(299, 299))
        # x = image.img_to_array(img)
        # x = preprocess_input(x)
        testing_data.append(np.array(img_data))
    np.save('test_data_299.npy', testing_data)
    return testing_data

def create_test_unlabeled_data():#lesaa el label 3ayz yt3ml 3aleh 1 hot encoding***
    testing_dataUnLabeled = []
    sortedFolders = []
    label = 1
    for folder in tqdm(os.listdir(labeledTest)):  # folder name 10 classes
        sortedFolders.append(folder)
    sortedFolders.sort()
    for folder in sortedFolders:
        folder_path = os.path.join(labeledTest, folder)
        for img in tqdm(os.listdir(folder_path)):# each folder has set of images
            img_path = os.path.join(folder_path, img)
            img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_data = img_data[:, :, [2, 1, 0]]
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))            
            testing_dataUnLabeled.append([np.array(img_data), label])  # el img el resized-label
        label = label + 1
    # shuffle(testing_dataUnLabeled)
    np.save('test_data_unlabel.npy', testing_dataUnLabeled)
    return testing_dataUnLabeled

def loadData():
    if (os.path.exists('train_data_299.npy')):
        train_data = np.load('train_data_299.npy', allow_pickle=True)
        print ("train data loaded")
    else:
        train_data = create_train_data()
        print("train data created")

    if (os.path.exists('test_data_299.npy')):
        test_data = np.load('test_data_299.npy', allow_pickle=True)
        print ("test data loaded")
    else:
        test_data = create_test_data()
        print("test data created")

    if (os.path.exists('test_data_unlabel.npy')):
      testDataUnLabel = np.load('test_data_unlabel.npy', allow_pickle=True)
      print("test un label data loaded")
    else:
      testDataUnLabel = create_test_unlabeled_data()
      print("test un label data created")

    X = np.array([i[0] for i in train_data])
    y = [i[1] for i in train_data]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0, shuffle=True,stratify=y)

    yTrain = [i - 1 for i in y_train]
    yTest = [i - 1 for i in y_test]

    y_train = np_utils.to_categorical(yTrain)
    y_test = np_utils.to_categorical(yTest)

    return X_train, X_test, y_train, y_test, test_data, testDataUnLabel

def main():
    X_train, X_test, y_train, y_test, test_data, testDataUnLabel = loadData()

    # model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,input_shape=None, pooling=None, classes=1000)
    base_model = InceptionResNetV2(include_top=False, weights='imagenet',input_shape=(IMG_SIZE, IMG_SIZE, 3))
    transfer_layer = base_model.get_layer('block8_10')
    # transfer_layer = base_model.layers[-1]
    conv_model = Model(inputs=base_model.input,
                   outputs=transfer_layer.output)

  
    
    new_model = Sequential()
    new_model.add(conv_model)
    new_model.add(Flatten())
    new_model.add(Dense(1024, activation='relu'))
    # new_model.add(Dropout(0.5))
    new_model.add(Dense(1024, activation='relu'))
    # new_model.add(Dropout(0.5))
    new_model.add(Dense(10, activation='softmax'))

    for layer in conv_model.layers:
        layer.trainable = False

    print("startt")
    new_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    # train the model on the new data for a few epochs
    # model.fit_generator(...)
    # model.summary()
    # new_model.fit(X_train, y_train, epochs=10, verbose=1, validation_data= (X_test, y_test))

    dataAugmentaion = ImageDataGenerator(rotation_range = 45, zoom_range = 0.20, 
    fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True, 
    width_shift_range = 0.1, height_shift_range = 0.1)

    new_model.fit_generator(dataAugmentaion.flow(X_train, y_train,batch_size = 64), validation_data = (X_test, y_test),
    steps_per_epoch = len(X_train) // 64, epochs = 100)

    # new_model.save('inceptionDataAug.h5')
    result =new_model.evaluate(X_test, y_test, verbose=1)
    print("Done testing")
    print("Test loss =", result[0])
    print("Test accuracy =", result[1] * 100)
    prediction = new_model.predict(test_data)
    predictionLabel= np.argmax(prediction,axis=1)
    predictionLabel=predictionLabel+1
    submitFile['Label']=predictionLabel
    
    TestID = []
    for img in tqdm(os.listdir(TEST_DIR)):
        TestID.append(img)
    submitFile['Id']= TestID 
    submitFile.to_csv("submitFile.csv",index=False)
    # new_model.save('inceptionModel.h5')
    
    prediction = new_model.predict(X_test)
    correct = 0
    for i in range(len(y_test)):
        prediction[i, :] = np.where(prediction[i, :] == max(prediction[i, :]), 1, 0)
        if np.array_equal(prediction[i], y_test[i]):
            correct += 1

    print("Overall Accuracy", (correct / len(y_test)) * 100, "%")
   
    # testDataUnLabel = create_test_unlabeled_data
    X = np.array([i[0] for i in testDataUnLabel])
    y = [i[1] for i in testDataUnLabel]
    yUnLabelTest = [i - 1 for i in y]
    y_un_label_test = np_utils.to_categorical(yUnLabelTest)

    predictionUnLabelTest = new_model.predict(X)
    correct = 0
    for i in range(len(y_un_label_test)):
        predictionUnLabelTest[i, :] = np.where(predictionUnLabelTest[i, :] == max(predictionUnLabelTest[i, :]), 1, 0)
        if np.array_equal(predictionUnLabelTest[i], y_un_label_test[i]):
            correct += 1

    print("Overall Accuracy un labeled data", (correct / len(y_un_label_test)) * 100, "%")
    
main()
