import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptronModel import *
import seaborn as sn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


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


# returns: X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, \
#            X4_test, labeled_Y_train, labeled_Y_test
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
    #region
    le.fit(target_train)
    le.fit(target_test)

    labeled_Y_train = le.transform(target_train)
    labeled_Y_train = labeled_Y_train.reshape(len(labeled_Y_train), 1)
    labeled_Y_train -= 1  # to map the classes to (-1, 0 ,1) like the return values of signum function

    labeled_Y_test = le.transform(target_test)
    labeled_Y_test = labeled_Y_test.reshape(len(labeled_Y_test), 1)
    labeled_Y_test -= 1
    #endregion

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
    print("confusion", confusion)
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

    # line
    x1 = min(feature1TestData) - 1  ## --> x
    print("W1", w1)
    print("W2", w2)
    y1 = ((-w2 * x1) - b) / w1  ## --> y

    x2 = max(feature1TestData) + 1  # --> x
    y2 = ((-w1 * x2) - b) / w2   # x1, X2 ## --> y
    point1 = [x1, x2]
    # point2 = [x1, x2[0]]
    point2 = [y1[0], y2[0]]
    print("point1", point1)
    print("point2", point2)

    plt.plot(point1, point2, color='red', linewidth=3)

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


# passing class1, class2 as "strings" not labels
def main(feature1, feature2, class1, class2, testFeature1, testFeature2, alpha, epochs, use_bias_bool, MSEthreshold):

    f1Train, f1Test, f2Train, f2Test = [], [], [], []

    epochs = int(epochs)
    use_bias_bool = bool(use_bias_bool)  # boolean
    alpha = float(alpha)
    MSEthreshold = float(MSEthreshold)

    if class2 < class1:
        class1, class2 = class2, class1
    #extracting features
    #region
    if feature1 == "X1":
        f1Train = X1_train
        f1Test = X1_test
    elif feature1 == "X2":
        f1Train = X2_train
        f1Test = X2_test
    elif feature1 == "X3":
        f1Train = X3_train
        f1Test = X3_test
    elif feature1 == "X4":
        f1Train = X4_train
        f1Test = X4_test

    if feature2 == "X1":
        f2Train = X1_train
        f2Test = X1_test
    elif feature2 == "X2":
        f2Train = X2_train
        f2Test = X2_test
    elif feature2 == "X3":
        f2Train = X3_train
        f2Test = X3_test
    elif feature2 == "X4":
        f2Train = X4_train
        f2Test = X4_test
#endregion

    #extracting classes
    #region
    if class1 == "Iris-setosa":
        newf1Test = f1Test[0:20]
        newf2Test = f2Test[0:20]
        newf1Train = f1Train[0:30]
        newf2Train = f2Train[0:30]

    elif class1 == "Iris-versicolor":
        newf1Test = f1Test[20:40]
        newf2Test = f2Test[20:40]
        newf1Train = f1Train[30:60]
        newf2Train = f2Train[30:60]

    elif class1 == "Iris-virginica":
        newf1Test = f1Test[40:]
        newf2Test = f2Test[40:]
        newf1Train = f1Train[60:]
        newf2Train = f2Train[60:]

    if class2 == "Iris-setosa":
        newf1Test=newf1Test.append(f1Test[0:20])
        newf2Test=newf2Test.append(f2Test[0:20])
        newf1Train=newf1Train.append(f1Train[0:30])
        newf2Train=newf2Train.append(f2Train[0:30])

    elif class2 == "Iris-versicolor":
        newf1Test=newf1Test.append(f1Test[20:40])
        newf2Test=newf2Test.append(f2Test[20:40])
        newf1Train = newf1Train.append(f1Train[30:60])
        newf2Train = newf2Train.append(f2Train[30:60])

    elif class2 == "Iris-virginica":
        newf1Test=newf1Test.append(f1Test[40:])
        newf2Test=newf2Test.append(f2Test[40:])
        newf1Train = newf1Train.append(f1Train[60:])
        newf2Train = newf2Train.append(f2Train[60:])
    #endregion

    train_X = np.array([newf1Train, newf2Train])
    test_X = np.array([newf1Test, newf2Test])

    c1Train = np.full(30, class1, dtype=object)
    c2Train = np.full(30, class2, dtype=object)

    c1Test = np.full(20, class1, dtype=object)
    c2Test = np.full(20, class2, dtype=object)

    train_Y = np.append(c1Train, c2Train)  #strings
    test_Y = np.append(c1Test, c2Test)  #strings

    le.fit(train_Y)
    le.fit(test_Y)

    train_Y = le.transform(train_Y)
    test_Y = le.transform(test_Y)

    class1 = 0
    class2 = 1

    train_X[0], train_X[1], train_Y = shuffle(train_X[0], train_X[1], train_Y)

    if class1 < class2:
        Model = perceptronModel(train_X, test_X, class1, class2, train_Y, test_Y, epochs, use_bias_bool, alpha)
    else:
        Model = perceptronModel(train_X, test_X, class2, class1, train_Y, test_Y, epochs, use_bias_bool, alpha)

    weights, bias = Model.training(MSEthreshold)
    if use_bias_bool == 0:
        bias = 0

    drawLine(feature1, feature2, newf1Test, newf2Test, test_Y, weights[0, 0], weights[0, 1], bias)

    accuracy, testing_predictions = Model.testing()

    print("Overall Accuracy is:", accuracy, "%")

    draw_confusion_matrix(test_Y, testing_predictions, class1, class2)

    yHat = Model.testInputData(testFeature1, testFeature2)

    print("Predicted class : ", le.inverse_transform([yHat]))
    print("-----------------------------------------")

