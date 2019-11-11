import numpy as np


class perceptronModel:
    # train_vector = (4*90)
    # test_vector = (4*60)
    def __init__(self, train_vector, test_vector, Y_train, Y_test, numOfHiddenLayers, numOfNeurons, alpha, epochs,
                 use_bias_bool, activationFunction):
        self.train_vector = train_vector
        self.test_vector = test_vector
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.epochs = epochs
        self.use_bias_bool = use_bias_bool
        self.weights = []
        self.bias = np.array([])
        self.alpha = alpha
        self.test_predictions = []
        self.accuracy = 0
        self.savedPredictons = []
        self.savedPredictonsTest = []
        self.numOfHiddenLayers = numOfHiddenLayers  # int
        self.numOfNeurons = numOfNeurons  # list shayla 3dd el neurons fe kol layer 3la 7sb el index
        self.activationFunction = activationFunction
        self.gradients = []

    def sigmoid(self, prediction):
        return 1 / (1 + np.exp(-prediction))

    def sigmoidPrime(self, prediction):
        dx = self.sigmoid(prediction)
        return dx * (1 - dx)

    def tanh(self, prediction):
        return (np.exp(prediction) - np.exp(-prediction)) / (np.exp(prediction) + np.exp(-prediction))

    def tanhPrime(self, prediction):
        dx = self.tanh(prediction)
        return 1 - pow(dx, 2)

    def initialization(self):
        for i in range(self.numOfHiddenLayers + 1):  # +1: hidden layers + output layer
            if i == 0:
                self.weights.append(np.random.rand(self.numOfNeurons[i], self.train_vector.shape[0]))
            else:
                self.weights.append(np.random.rand(self.numOfNeurons[i], self.numOfNeurons[i - 1]))

            # print("Weights", self.weights[i].shape)
        self.weights = np.asarray(self.weights)
        if (self.use_bias_bool == 1):
            self.bias = np.array(np.random.rand(self.numOfHiddenLayers + 1, 1))

    def firstForward(self):
        for HL in range(self.numOfHiddenLayers + 1):
            if self.use_bias_bool == 1:
                if HL == 0:
                    prediction = np.dot(self.train_vector.T, self.weights[HL].T) + self.bias[HL]
                else:
                    prediction = np.dot(self.savedPredictons[HL - 1], self.weights[HL].T) + self.bias[HL]

            else:
                if HL == 0:
                    prediction = np.dot(self.train_vector.T, self.weights[HL].T)
                else:
                    e = np.asarray(self.savedPredictons[HL - 1])
                    prediction = np.dot(e, self.weights[HL].T)

            self.savedPredictons.append(prediction)

    def backward(self):  # calculate gradient for each layer

        # calculate error at the output layer
        if self.activationFunction == "Sigmoid":
            outputLayerActivation = self.sigmoid(self.savedPredictons[-1])
        else:
            outputLayerActivation = self.tanh(self.savedPredictons[-1])
        output_e = self.Y_train - outputLayerActivation
        output_E = np.power(output_e, 2) * 0.5  # el mafrod vector wala rqm

        # calculate output gradient:
        # partial E / partial W_kj

        if self.activationFunction == "Sigmoid":
            self.outputLayerGradient = output_e * self.sigmoidPrime(outputLayerActivation) * self.sigmoid(
                self.savedPredictons[-1])
        else:
            self.outputLayerGradient = output_e * self.tanhPrime(outputLayerActivation) * self.tanh(
                self.savedPredictons[-1])

        # self.outputLayerGradient = outputLocalGradient * self.savedPredictons[-1]
        # hwa msh el mafrod *Zj y3ne el prediction b3d mayd5ol 3la el activation?? (e fl f prime fl zj)
        self.gradients.append(self.outputLayerGradient)
        index = 0

        #####
        # calculate hidden layers gradient:
        for i in reversed(range(self.numOfHiddenLayers)):  # msh hanktb -1 3ashan hya mabtbd2sh ble mktob
            if self.activationFunction == "Sigmoid":
                hiddenLayerGradient = self.sigmoidPrime(self.savedPredictons[i]) * np.dot(self.gradients[index],
                                                                                          self.weights[i + 1])
            else:
                hiddenLayerGradient = self.tanhPrime(self.savedPredictons[i]) * np.dot(self.gradients[index],
                                                                                       self.weights[i + 1])

            self.gradients.append(hiddenLayerGradient)
            index += 1

    def secondForward(self):  # update weights

        self.gradients.reverse()
        for i in range(len(self.gradients)):
            if i == 0:

                self.weights[i] = self.weights[i] + (self.alpha * np.dot(self.gradients[i].T, self.train_vector.T))
                if self.use_bias_bool == 1:
                    self.bias[i] = self.bias[i] + (self.alpha * np.dot(self.gradients[i].T, self.train_vector.T))
            else:
                if self.activationFunction == "Sigmoid":
                    Zj = self.sigmoid(self.savedPredictons[i - 1])  # leeh prime??
                else:
                    Zj = self.tanh(self.savedPredictons[i - 1])

                self.weights[i] = self.weights[i] + (self.alpha * np.dot(self.gradients[i].T, Zj))
                if self.use_bias_bool == 1:
                    self.bias[i] = self.bias[i] + self.alpha * np.dot(self.gradients[i].T, Zj.T)

    def training(self):
        self.initialization()
        for i in range(self.epochs):
            self.firstForward()
            self.backward()
            self.secondForward()
            self.savedPredictons.clear()
            self.gradients.clear()

    def testing(self):
        correct = 0
        print("TESTING...")
        for HL in range(self.numOfHiddenLayers + 1):
            if self.use_bias_bool == 1:
                if HL == 0:
                    prediction = np.dot(self.test_vector.T, self.weights[HL].T) + self.bias[HL]
                else:
                    prediction = np.dot(self.savedPredictonsTest[HL - 1], self.weights[HL].T) + self.bias[HL]

            else:
                if HL == 0:
                    prediction = np.dot(self.test_vector.T, self.weights[HL].T)
                else:
                    e = np.asarray(self.savedPredictonsTest[HL - 1])
                    prediction = np.dot(e, self.weights[HL].T)

            self.savedPredictonsTest.append(prediction)
        if self.activationFunction == "Sigmoid":
            testRes = self.sigmoid(self.savedPredictonsTest[-1])
        else:
            testRes = self.tanh(self.savedPredictonsTest[-1])

        print("RESULT", testRes)
        for i in range(testRes.shape[0]):
            testRes[i, :] = np.where(testRes[i, :] == max(testRes[i, :]), 1, 0)
            if np.array_equal(testRes[i, :], self.Y_test[i, :]):
                correct += 1
        print("RESULT", testRes)

        self.accuracy = (correct / testRes.shape[0]) * 100

        return self.accuracy
