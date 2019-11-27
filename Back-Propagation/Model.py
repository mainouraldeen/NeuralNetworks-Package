# import keras.utils as np_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import tensorflow.keras.utils as np_utils
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from perceptronModel import *


def readFile():
    alllines = open('IrisData.txt', 'r').readlines()
    data = np.matrix([line.replace('\n', '').split(',')[0:5] for line in alllines])
    data = np.delete(data, 0, axis=0)
    df = pd.DataFrame(data=data.flatten().reshape(150, 5))
    return df


def load_data(data):
    X1_train = data.iloc[:30, 0]
    X1_train = pd.concat([X1_train, data.iloc[50:80, 0]])
    X1_train = pd.concat([X1_train, data.iloc[100:130, 0]])

    X1_test = data.iloc[30:50, 0]
    X1_test = pd.concat([X1_test, data.iloc[80:100, 0]])
    X1_test = pd.concat([X1_test, data.iloc[130:, 0]])

    X2_train = data.iloc[:30, 1]
    X2_train = pd.concat([X2_train, data.iloc[50:80, 1]])
    X2_train = pd.concat([X2_train, data.iloc[100:130, 1]])

    X2_test = data.iloc[30:50, 1]
    X2_test = pd.concat([X2_test, data.iloc[80:100, 1]])
    X2_test = pd.concat([X2_test, data.iloc[130:, 1]])

    X3_train = data.iloc[:30, 2]
    X3_train = pd.concat([X3_train, data.iloc[50:80, 2]])
    X3_train = pd.concat([X3_train, data.iloc[100:130, 2]])

    X3_test = data.iloc[30:50, 2]
    X3_test = pd.concat([X3_test, data.iloc[80:100, 2]])
    X3_test = pd.concat([X3_test, data.iloc[130:, 2]])

    X4_train = data.iloc[:30, 3]
    X4_train = pd.concat([X4_train, data.iloc[50:80, 3]])
    X4_train = pd.concat([X4_train, data.iloc[100:130, 3]])

    X4_test = data.iloc[30:50, 3]
    X4_test = pd.concat([X4_test, data.iloc[80:100, 3]])
    X4_test = pd.concat([X4_test, data.iloc[130:, 3]])

    Y_train = data.iloc[:30, 4]
    Y_train = pd.concat([Y_train, data.iloc[50:80, 4]])
    Y_train = pd.concat([Y_train, data.iloc[100:130, 4]])

    Y_test = data.iloc[30:50, 4]
    Y_test = pd.concat([Y_test, data.iloc[80:100, 4]])
    Y_test = pd.concat([Y_test, data.iloc[130:, 4]])
    return X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, Y_train, Y_test


def dataPreprocessing():
    df = readFile()
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, target_train, target_test = load_data(
        df)

    X1_train = X1_train.astype(float)
    X2_train = X2_train.astype(float)
    X3_train = X3_train.astype(float)
    X4_train = X4_train.astype(float)

    X1_test = X1_test.astype(float)
    X2_test = X2_test.astype(float)
    X3_test = X3_test.astype(float)
    X4_test = X4_test.astype(float)

    # hot encoding for output
    # region
    le.fit(target_train)
    le.fit(target_test)

    labeled_Y_train = le.transform(target_train)
    labeled_Y_train = labeled_Y_train.reshape(len(labeled_Y_train), 1)

    labeled_Y_test = le.transform(target_test)
    labeled_Y_test = labeled_Y_test.reshape(len(labeled_Y_test), 1)

    labeled_Y_train = np_utils.to_categorical(labeled_Y_train, 3)
    labeled_Y_test = np_utils.to_categorical(labeled_Y_test, 3)
    # endregion

    return X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, \
           X4_test, labeled_Y_train, labeled_Y_test


# public
# region
le = preprocessing.LabelEncoder()
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, labeled_Y_train, labeled_Y_test = dataPreprocessing()


# endregion


def draw_confusion_matrix(y_test, y_predict):
    lbls = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    confusion = confusion_matrix(y_test, y_predict)
    print("Confusion Matrix:")
    print(confusion)

    df_cm = pd.DataFrame(list(confusion), index=[i for i in lbls], columns=[i for i in lbls])

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt._show()


def main(numOfHiddenLayers, numOfNeurons, alpha, epochs, use_bias_bool, activationFunction):
    epochs = int(epochs)
    numOfHiddenLayers = int(numOfHiddenLayers)
    numOfNeurons = np.array(numOfNeurons)
    numOfNeurons = numOfNeurons.astype(int)
    numOfNeurons = np.append(numOfNeurons, 3)
    use_bias_bool = int(use_bias_bool)  # boolean
    alpha = float(alpha)

    train_X = np.array([X1_train, X2_train, X3_train, X4_train])
    test_X = np.array([X1_test, X2_test, X3_test, X4_test])

    Model = perceptronModel(train_X, test_X, labeled_Y_train, labeled_Y_test, numOfHiddenLayers, numOfNeurons, alpha,
                            epochs, use_bias_bool, activationFunction)

    Model.training()

    accuracy, y_prediction = Model.testing()
    print("Overall Accuracy is:", accuracy, "%")

    # 3shan trg3hom tany l arkam
    y_prediction = [np.argmax(y, axis=None, out=None) for y in Model.savedPredictonsTest[-1]]
    y_test = [np.argmax(y, axis=None, out=None) for y in labeled_Y_test]

    draw_confusion_matrix(y_test, y_prediction)
