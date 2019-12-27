#Menna Last Code
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from random import shuffle
import os
import cv2
import numpy as np
from tqdm import tqdm  # Or from tqdm import tqdm if not jupyter notebook
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing import image
from PIL import Image
import time
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.layers import Input
from keras.models import Model,Sequential
from keras import optimizers
from keras.utils import np_utils
from sklearn.utils import shuffle
import sys
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from google.colab import files
import pandas as pd

# public
# region

TRAIN_DIR ='/content/drive/My Drive/neural data/train'
TEST_DIR = '/content/drive/My Drive/neural data/test'
#weight_Path='/content/drive/My Drive/return 0/neural data/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
submitFile= pd.read_csv('/content/drive/My Drive/neural data/submit.csv',keep_default_na=True)
IMG_SIZE = 224
LR = 0.001
MODEL_NAME = 'identify-places-cnn'
num_classes=10
# endregion

folders=[]
for folder in tqdm(os.listdir(TRAIN_DIR)):
  folders.append(folder)

folders.sort()
def create_train_data():
    training_data = []
    label = 1
    for folder in folders:   # folder name 10 classes
        folder_path = os.path.join(TRAIN_DIR, folder)
        for img in os.listdir(folder_path):  # each folder has set of images
            img_path = os.path.join(folder_path, img)
            img_data = cv2.imread(img_path, 0)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            training_data.append([np.array(img_data), label])  # el img el resized-label
        label = label + 1
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    print('done training')
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
       
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        testing_data.append(np.array(img_data))
    
    np.save('test_data.npy', testing_data)  
    print('done testing')
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

X = np.array([i[0] for i in train_data])
y = [i[1]-1 for i in train_data]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True,stratify=y)
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)



def resnet():
    
    restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    restnet = Model(restnet.input, output=output)
    for layer in restnet.layers:
        layer.trainable = False
  
  
    model = Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu'))#,input_dim=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    #model.load_weights(weight_Path,by_name=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    #model.summary()
    '''history = model.fit(X_train,y_train,epochs=10,verbose=1)
    predictions=model.predict(X_test)
    result = model.evaluate(X_test, y_test, verbose=1)
    print("Done testing")
    print("Test loss =", result[0])
    print("Test accuracy =", result[1] * 100)'''

    ###############Fine tuning################
    restnet.trainable = True
    set_trainable = False
    for layer in restnet.layers:
        if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    layers = [(layer, layer.name, layer.trainable) for layer in restnet.layers]
    model_finetuned = Sequential()
    model_finetuned.add(restnet)
    model_finetuned.add(Dense(512, activation='relu'))#, input_dim=input_shape))
    #model_finetuned.add(Dropout(0.3))
    model_finetuned.add(Dense(512, activation='relu'))
    #model_finetuned.add(Dropout(0.3))
    model_finetuned.add(Dense(10, activation='sigmoid'))
    model_finetuned.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])
    model_finetuned.summary()
    history = model_finetuned.fit(X_train,y_train,epochs=10,verbose=1)
    predictions=model_finetuned.predict(X_test)
    result =model_finetuned.evaluate(X_test, y_test, verbose=1)
    print("Done testing")
    
    print("Test loss =", result[0])
    print("Test accuracy =", result[1] * 100)
    model_finetuned.save('resnetModelFineTuned.h5')
    #files.download('resnetModelFineTuned.h5')
    predictionstest=model_finetuned.predict(test_data)
    predictionLabel= np.argmax(predictionstest,axis=1)
    predictionLabel=predictionLabel+1
    submitFile['Label']=predictionLabel
    TestID=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        TestID.append(img)
    submitFile['Id']= TestID  
    submitFile.to_csv("submitFile.csv",index=False)
    #files.download("submitFile.csv")
   
resnet()