import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
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
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, \
    X4_test, target_train, target_test = load_data(df)

    X1_train = X1_train.astype(float)
    X2_train = X2_train.astype(float)
    X3_train = X3_train.astype(float)
    X4_train = X4_train.astype(float)

    X1_test = X1_test.astype(float)
    X2_test = X2_test.astype(float)
    X3_test = X3_test.astype(float)
    X4_test = X4_test.astype(float)

    # I think we needn't to label Y here
    # region
    le.fit(target_train)
    le.fit(target_test)

    labeled_Y_train = le.transform(target_train)
    labeled_Y_train = labeled_Y_train.reshape(len(labeled_Y_train), 1)
    labeled_Y_train -= 1  # to map the classes to (-1, 0 ,1) like the return values of signum function

    labeled_Y_test = le.transform(target_test)
    labeled_Y_test = labeled_Y_test.reshape(len(labeled_Y_test), 1)
    labeled_Y_test -= 1
    # endregion

    return X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, \
           X4_test, labeled_Y_train, labeled_Y_test


# public
# region
le = preprocessing.LabelEncoder()
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, labeled_Y_train, labeled_Y_test = dataPreprocessing()


# endregion


def draw_confusion_matrix(y_test, y_predict, c1, c2):
    lbls = []

    if c1 == -1:
        lbls.append("Iris-setosa")
    elif c1 == 0:
        lbls.append("Iris-versicolor")
    elif c1 == 1:
        lbls.append("Iris-virginica")

    if c2 == -1:
        lbls.append("Iris-setosa")
    elif c2 == 0:
        lbls.append("Iris-versicolor")
    elif c2 == 1:
        lbls.append("Iris-virginica")

    confusion = confusion_matrix(y_test, y_predict)
    print("confusion")
    print(confusion)
    #
    # print("lbls")
    # print(lbls)

    df_cm = pd.DataFrame(list(confusion), index=[i for i in lbls],
                         columns=[i for i in lbls])

    # print("\ndf_cm", df_cm)
    # print("df_cm shape", df_cm.shape)
    # print("df_cm type", type(df_cm))

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt._show()


def drawLine(f1, f2, feature1TestData, feature2TestData, test_Y, w1, w2, b):
    color = ['grey', 'blue']
    # test_Y += 1

    plt.scatter(feature1TestData, feature2TestData, c=test_Y, cmap=matplotlib.colors.ListedColormap(color))
    plt.xlabel(f1, fontsize=20)
    plt.ylabel(f2, fontsize=20)

    cb = plt.colorbar()

    loc = np.arange(0, max(test_Y), max(test_Y) / float(len(color)))
    cb.set_ticks(loc)
    lbls = le.inverse_transform(np.unique(test_Y))
    cb.set_ticklabels(lbls)

    pointX, pointY = [], []
    x2 = min(feature1TestData) - 5
    x1 = ((-w2 * x2) - b) / w1

    pointX.append(x2)
    pointY.append(x1[0])

    x1 = max(feature1TestData) + 5
    x2 = ((-w1 * x1) - b) / w2

    pointX.append(x1)
    pointY.append(x2[0])

    plt.plot(pointX, pointY, color='green', linewidth=2)

    plt._show()


def drawIrisData():
    #################x1,X2#################
    x1 = X1_train.append(X1_test)
    x2 = X2_train.append(X2_test)
    label = np.append(labeled_Y_train + 1, labeled_Y_test + 1)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x1, x2, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X2', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()

    #################x1,X3#################
    x1 = X1_train.append(X1_test)
    x3 = X3_train.append(X3_test)
    label = np.append(labeled_Y_train + 1, labeled_Y_test + 1)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x1, x3, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X3', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()

    #################x1,X4#################
    x1 = X1_train.append(X1_test)
    x4 = X4_train.append(X4_test)
    label = np.append(labeled_Y_train + 1, labeled_Y_test + 1)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x1, x4, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X4', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()

    #################x2,X3#################
    x2 = X2_train.append(X2_test)
    x3 = X3_train.append(X3_test)
    label = np.append(labeled_Y_train + 1, labeled_Y_test + 1)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x2, x3, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X2', fontsize=20)
    plt.ylabel('X3', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()

    #################x2,X4#################
    x2 = X2_train.append(X2_test)
    x4 = X4_train.append(X4_test)
    label = np.append(labeled_Y_train + 1, labeled_Y_test + 1)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x2, x4, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X2', fontsize=20)
    plt.ylabel('X4', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()

    #################x3,X4#################
    x3 = X3_train.append(X3_test)
    x4 = X4_train.append(X4_test)
    label = np.append(labeled_Y_train + 1, labeled_Y_test + 1)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x3, x4, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X3', fontsize=20)
    plt.ylabel('X4', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()


def main(numOfHiddenLayers, numOfNeurons, alpha, epochs, use_bias_bool, activationFunction):
    epochs = int(epochs)
    numOfHiddenLayers = int(numOfHiddenLayers)
    numOfNeurons = np.array(numOfNeurons)
    numOfNeurons = numOfNeurons.astype(int)
    use_bias_bool = int(use_bias_bool)  # boolean
    alpha = float(alpha)
    print("Bias", use_bias_bool)
    print("activationFunction", activationFunction)
    Model = perceptronModel([X1_train, X2_train, X3_train, X4_train], [X1_test, X2_test, X3_test, X4_test],
                            labeled_Y_train, labeled_Y_test, numOfHiddenLayers, numOfNeurons, alpha, epochs,
                            use_bias_bool, activationFunction)
    if use_bias_bool == 0:
        bias = 0

    Model.firstForward()

    # accuracy, testing_predictions = Model.testing()

    # print("Overall Accuracy is:", accuracy, "%")

    # draw_confusion_matrix(test_Y, testing_predictions, class1, class2)

    print("-----------------------------------------")
