import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing


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


data = pd.read_csv('IrisData.csv')
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, target_train, target_test = load_data(
    data)

le = preprocessing.LabelEncoder()
le.fit(target_train)
le.fit(target_test)
labeled_Y_train = le.transform(target_train)
labeled_Y_test = le.transform(target_test)


def DrawIrisData():
    # x1,X2
    x1 = X1_train.append(X1_test)
    x2 = X2_train.append(X2_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x1, x2, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X2', fontsize=20)
    plt._show()

    # x1,X3
    x1 = X1_train.append(X1_test)
    x3 = X3_train.append(X3_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x1, x3, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X3', fontsize=20)
    plt._show()

    # x1,X4
    x1 = X1_train.append(X1_test)
    x4 = X4_train.append(X4_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x1, x4, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X4', fontsize=20)
    plt._show()

    # x2,X3
    x2 = X2_train.append(X2_test)
    x3 = X3_train.append(X3_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x2, x3, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X2', fontsize=20)
    plt.ylabel('X3', fontsize=20)
    plt._show()

    # x2,X4
    x2 = X2_train.append(X2_test)
    x4 = X4_train.append(X4_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x2, x4, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X2', fontsize=20)
    plt.ylabel('X4', fontsize=20)
    plt._show()

    # x3,X4
    x3 = X3_train.append(X3_test)
    x4 = X4_train.append(X4_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x3, x4, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X3', fontsize=20)
    plt.ylabel('X4', fontsize=20)
    plt._show()


def perceptron():
    print("doneeeeeeeeeeeeeeee algo")


def signum(prediction):
    return 1 if prediction > 0 else -1

    # 3ndy vector ll weights l kol layer msh l kol input
    # w ana 3ndy one layer: only the output layer
    # current layer: output layer (only one neuron)


def perceptron_model(X, Y, alpha, epochs):  # assume Y: numbers not words

    w = np.random.rand(1, 2)  # W[#neurons of the current layer, #neurons of the previous layer]
    b = np.zeros(1, 1)  # b[#neurons of the current layer, 1]

    for i in range(epochs):
        # prediction = np.dot(w1, X1) + np.dot(w2, X2) + b  # ?np.dot wala *

        prediction = np.dot(w, X) + b  # ?np.dot wala *
        yHat = signum(prediction)  # ?values of signum
        error = Y - yHat  # ? y hat victor wala la
        w = w + (alpha * error * X)  # anhy x ??
        b = b + (alpha * error)  # adrb f eh!
    return w, b


def DrawData():
    DrawIrisData()


def main():
    '''
    le = preprocessing.LabelEncoder()
    le.fit(target_train)
    le.fit(target_test)
    labeled_Y_train = le.transform(target_train)
    labeled_Y_test = le.transform(target_test)

    alpha = 0.001  # user input
    epochs = 200  # user input

    # hl 2 loops: wa7da tmshy 3l dataset (bb3t element element) w wa7da ll training??
    for i in range(len(X1_train)):
        W, b = perceptron_model([X1_train[i], X2_train[i]], labeled_Y_train[i], alpha, epochs)

    # msh 3arfa hn-test ezay, hnb3t X_train ezay? element element wala vector??
    '''


main()

'''# Loading data
inputFile = open("IrisData.txt", "r+")
fileLines = inputFile.readlines()
numLines = sum(1 for line in open("IrisData.txt")) - 1
dataMatrix = np.empty([numLines,5])

for line in fileLines:
    data = re.split('[,\\n]', line)
    dataMatrix+=data'''
