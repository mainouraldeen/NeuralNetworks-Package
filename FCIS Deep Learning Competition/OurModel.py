# Nada

# # region OUR OWN MODEL
#
import os
import cv2
import numpy as np
import keras
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout, SpatialDropout2D
from keras.models import Sequential
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
# # public
# # region
TRAIN_DIR = '/gdrive/My Drive/return 0/neural data/train'
TEST_DIR = '/gdrive/My Drive/return 0/neural data/test'
IMG_SIZE = 256
LR = 0.001
MODEL_NAME = 'identify-places-cnn'
#
# # end region
#
def create_train_data():  # lesaa el label 3ayz yt3ml 3aleh 1 hot encoding***
    print("Inside train");
    training_data = []
    label = 1
    for folder in os.listdir(TRAIN_DIR):  # folder name 10 classes
        folder_path = os.path.join(TRAIN_DIR, folder)
        for img in os.listdir(folder_path):  # each folder has set of ima0oooooooooooooooooooooooooges
            print("train ", img);
            print("label", label)
            img_path = os.path.join(folder_path, img)
            img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_data = img_data[:, :, [2, 1, 0]]  # Make it RGB
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), label])  # el img el resized-label
        label = label + 1
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    print("done create train");
    return training_data


def create_test_data():
    print("Inside test");
    testing_data = []
    for img in os.listdir(TEST_DIR):
        print("test ", img);
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = img_data[:, :, [2, 1, 0]]  # Make it RGB

        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append(np.array(img_data))
    np.save('test_data.npy', testing_data)
    print("done create test");
    return testing_data

# region load data
if os.path.exists('train_data.npy'):
    print("train_data.npy exists")
    train_data = np.load('train_data.npy', allow_pickle=True)
else:
    print("train_data.npy doesn't exist !!!")
    train_data = create_train_data()

if os.path.exists('test_data.npy'):
    print("test_data.npy exists")
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    print("test_data.npy doesn't exist !!!")
    test_data = create_test_data()
# endregion


X = np.array([i[0] for i in train_data])
y = [i[1] - 1 for i in train_data]


X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y,test_size=0.3, random_state=0, shuffle=True)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

input_X = X_train
test_input = X_test

model = Sequential()

# region########################### Model Architecture: ###########################

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))  #3) 99.05666
model.add(BatchNormalization())##

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())##

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())##

model.add(Dropout(0.2))  #5) 99.0999
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

# fully connected layer
model.add(Dense(256, activation='relu'))

# output layer
model.add(Dense(10, activation='softmax'))
# endregion

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adadelta', metrics=['accuracy'])
print(input_X.shape)
print((Y_train))
model.fit(input_X, Y_train, batch_size=256, epochs=20, verbose=2)
print("Model compiled")

# Test:
print("\nTesting...")
result = model.evaluate(X_test, Y_test, verbose=0)
print("Done testing")

print("Test loss =", result[0])
print("Test accuracy =", result[1]*100)

model.summary()
predictions = model.predict(test_data)
predictionLabel = np.argmax(predictions,axis=1)
predictionLabel = predictionLabel + 1
submitFile['Label'] = predictionLabel
submitFile.to_csv("submitFile.csv",index=False)
files.download("submitFile.csv")

# plot_model(model, to_file='model.png')

