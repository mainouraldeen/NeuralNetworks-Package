import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
import tensorflow as tf
from random import shuffle
import os
import cv2
import numpy as np
from tqdm import tqdm  # Or from tqdm import tqdm if not jupyter notebook
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

# public
# region

TRAIN_DIR = 'tiny_train'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'identify-places-cnn'


# endregion

def create_train_data():
    training_data = []
    label = 1
    for folder in tqdm(os.listdir(TRAIN_DIR)):  # folder name 10 classes
        folder_path = os.path.join(TRAIN_DIR, folder)
        for img in os.listdir(folder_path):  # each folder has set of images
            img_path = os.path.join(folder_path, img)
            img_data = cv2.imread(img_path, 0)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), label])  # el img el resized-label
        label = label + 1
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append(np.array(img_data))
    np.save('test_data.npy', testing_data)
    return testing_data


if (os.path.exists('train_data.npy')):
    train_data = np.load('train_data.npy', allow_pickle=True)
else:
    train_data = create_train_data()

if (os.path.exists('test_data.npy')):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = create_test_data()

X = np.array([i[0] for i in train_data])
y = [i[1] for i in train_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=False)


def dataAugmentation():
    image_file_name = 'pool.jpg'
    img = cv2.imread(image_file_name, 1)
    img = img[:, :, [2, 1, 0]]
    img = cv2.resize(img, (300, 300))
    # plt.axis("off")
    # plt.imshow(img)
    # plt.show()

    # ele fo2 da sha8al msh 3arfa b3d m3mlt augmentation asave aw a3rd el sora!
    image_string = tf.read_file(image_file_name)
    img = tf.image.decode_jpeg(image_string, channels=3)
    # img=tf.image.convert_image_dtype(img,dtype=tf.float32)
    img = 2 * (img / 255.0) - 1.0
    # im = tf.expand_dims(img, 0)
    imagee = tf.image.flip_left_right(img)

    filename = 'savedImage.jpg'
    tf.write_file(filename, imagee)

    # imagee = 2 * (imagee / 255.0) - 1.0
    # show_image(img, imagee, "flip")


def dataPreprocessing():
    pass


def show_image(original_image, augmented_image, title):
    fig = plt.figure()
    fig.suptitle(title)

    original_plt = fig.add_subplot(1, 2, 1)  # (1 row 2 cols (sorten), 1 awl sora kaza)

    original_plt.set_title('original image')
    original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)

    original_plt.imshow(np.real(original_image))

    augmented_plt = fig.add_subplot(1, 2, 2)
    augmented_plt.set_title('augmented image')
    augmented_image = tf.image.convert_image_dtype(augmented_image, dtype=tf.float32)

    augmented_plt.imshow(augmented_image)
    plt.show(block=True)


def main():
    pass


main()
