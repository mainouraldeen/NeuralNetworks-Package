import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics


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


def readFile():
    alllines = open('IrisData.txt', 'r').readlines()
    data = np.matrix([line.replace('\n', '').split(',')[0:5] for line in alllines])
    data = np.delete(data, 0, axis=0)
    df = pd.DataFrame(data=data.flatten().reshape(150, 5))
    return df


def makeCleanData():
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

    le = preprocessing.LabelEncoder()
    le.fit(target_train)
    le.fit(target_test)

    labeled_Y_train = le.transform(target_train)
    labeled_Y_train = labeled_Y_train.reshape(len(labeled_Y_train), 1)
    labeled_Y_train -= 1  # to map the classes to (-1, 0 ,1) like the return values of signum function

    labeled_Y_test = le.transform(target_test)
    labeled_Y_test = labeled_Y_test.reshape(len(labeled_Y_test), 1)
    labeled_Y_test -= 1

    return X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, \
           X4_test, labeled_Y_train, labeled_Y_test


# public
# region
# data = pd.read_csv('IrisData.txt')
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, labeled_Y_train, labeled_Y_test = makeCleanData()
'''
le = preprocessing.LabelEncoder()
le.fit(target_train)
le.fit(target_test)
labeled_Y_train = le.transform(target_train)
labeled_Y_test = le.transform(target_test)
'''


# endregion

def testing(f1Data, f2Data, c1, c2, w1, w2, b):
    correct = 0
    f1Test, f2Test, c1Test, c2Test, classTest = [], [], [], [], []

    if c1 == -1:
        c1Test = labeled_Y_test[0:20]
        f1Test = f1Data[0:20]
        f2Test = f2Data[0:20]
    elif c1 == 0:
        c1Test = labeled_Y_test[20:40]
        f1Test = f1Data[20:40]
        f2Test = f2Data[20:40]
    elif c1 == 1:
        c1Test = labeled_Y_test[40:60]
        f1Test = f1Data[40:60]
        f2Test = f2Data[40:60]

    if c2 == -1:
        c2Test = labeled_Y_test[0:20]
        f1Test = f1Test.append(f1Data[0:20])
        f2Test = f2Test.append(f2Data[0:20])
    elif c2 == 0:
        c2Test = labeled_Y_test[20:40]
        f1Test = f1Test.append(f1Data[20:40])
        f2Test = f2Test.append(f2Data[20:40])
    elif c2 == 1:
        c2Test = labeled_Y_test[40:60]
        f1Test = f1Test.append(f1Data[40:60])
        f2Test = f2Test.append(f2Data[40:60])

    classTest = np.append(c1Test, c2Test)
    classTest.sort()

    for i in range(40):
        prediction = np.dot(w1, f1Test[i]) + np.dot(w2, f2Test[i]) + b
        yHat = signum(prediction)
        error = classTest[i] - yHat
        if error == 0:
            correct += 1

    accuracy = (correct / 40) * 100
    return accuracy


def drawLine(feature1Data, feature2Data, classData, w1, w2, b):
    colors = ['grey', 'blue', 'purple']
    plt.scatter(feature1Data, feature2Data, c=classData, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X2', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(classData), max(classData) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    # line
    x2 = 0
    x1 = ((-w2 * x2) - b) / w1  # X1, 0
    point1 = [x1, 0]
    x1 = 0
    x2 = ((-w1 * x1) - b) / w2  # 0, X2
    point2 = [0, x2]

    plt.plot(point1, point2, color='red', linewidth=3)
    plt.plot()
    plt._show()


def DrawIrisData():
    #################x1,X2#################
    x1 = X1_train.append(X1_test)
    x2 = X2_train.append(X2_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x1, x2, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X2', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()
    drawLine(x1, x2, label)
    #################x1,X3#################
    x1 = X1_train.append(X1_test)
    x3 = X3_train.append(X3_test)
    label = np.append(labeled_Y_train, labeled_Y_test)
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
    label = np.append(labeled_Y_train, labeled_Y_test)
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
    label = np.append(labeled_Y_train, labeled_Y_test)
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
    label = np.append(labeled_Y_train, labeled_Y_test)
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
    label = np.append(labeled_Y_train, labeled_Y_test)
    colors = ['grey', 'blue', 'purple']

    plt.scatter(x3, x4, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X3', fontsize=20)
    plt.ylabel('X4', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(['C1-setosa', 'C2-versicolor', 'C3-virginica'])
    plt._show()


def perceptron():
    print("doneeeeeeeeeeeeeeee algo")


def signum(prediction):
    if prediction > 0:
        return 1
    else:
        return -1


def perceptron_model(X, Y, alpha, epochs, bias):
    w = list(np.random.rand(1, 2))
    b = 0  # np.random.rand(1, 1)
    for i in range(int(epochs)):
        for j in range(len(X)):
            res = np.dot(w[0], X[:, j].reshape(2, 1))  # .reshape(1, 1)
            res = res[0]
            print("res", res)

            print("res type", type(res))

            prediction = (res + b)
            prediction = prediction.reshape(1, 1)
            # prediction = prediction[0]
            print("pre = ", prediction)
            y_hat = signum(prediction)
            print("pre type", type(prediction))
            print("pre shape", prediction.shape)
            print("--------------------------------")
            if y_hat != Y[j]:
                error = Y[j] - y_hat
                w[0] = w[0] + (alpha * error.reshape(1, 1) * X[:, j].reshape(2, 1))
                b = b + (alpha * error)

    return w, b


def main(feature1, feature2, class1, class2, alpha, epochs, bias):
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, \
    X4_test, labeled_Y_train, labeled_Y_test = makeCleanData()
    f1Train, f1Test, f2Test, f2Test = [], [], [], []

    if feature1 == "X1":  # initialize lists???
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

    if feature2 == "X1":  # initialize lists???
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

    # input = np.array([f1Train, f2Train])
    # W, b = perceptron_model(input, labeled_Y_train, alpha, epochs, bias)
    # W = np.array(W)

    # don't call this fn now
    '''if class1 < class2:
        testing(f1Test, f2Test, class1, class2, 1, 1, bias)
    else:
        testing(f1Test, f2Test, class2, class1, 1, 1, bias)'''

# main()
