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
        self.numOfHiddenLayers = numOfHiddenLayers  # int
        self.numOfNeurons = numOfNeurons  # list shayla 3dd el neurons fe kol layer 3la 7sb el index
        self.activationFunction = activationFunction
        self.gradients = []

    def sigmoid(self, prediction):
        return 1 / 1 + np.exp(-prediction)

    def sigmoidPrime(self, prediction):
        dx = self.sigmoid(prediction)
        return dx * (1 - dx)

    def tanh(self, prediction):
        return (np.exp(prediction) - np.exp(-prediction)) / (np.exp(prediction) - np.exp(-prediction))

    def tanhPrime(self, prediction):
        dx = self.tanh(prediction)
        return 1 - pow(dx, 2)

    def firstForward(self):
        # initialize random weights matrices
        print("len", self.train_vector.shape)
        for i in range(self.numOfHiddenLayers + 1):  # +1: hidden layers + output layer
            if i == 0:
                self.weights.append(np.random.rand(self.numOfNeurons[i], self.train_vector.shape[0]))
            else:
                self.weights.append(np.random.rand(self.numOfNeurons[i], self.numOfNeurons[i - 1]))

            print("Weights", self.weights[i].shape)
        self.weights = np.asarray(self.weights)
        self.bias = np.array(np.random.rand(self.numOfHiddenLayers + 1, 1))
        # print("bias shape", self.bias.shape)

        # Algo:(
        for HL in range(self.numOfHiddenLayers + 1):
            # print("HL", HL)
            # print("self.train_vector.T", self.train_vector.T.shape)
            # print("self.weights[HL].T", self.weights[HL].T.shape)

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


            # print("predi", prediction.shape)
            self.savedPredictons.append(prediction)
            # print(len(self.savedPredictons))

        # self.savedActivations = np.asarray(self.savedActivations)
        # print("a5er wa7ed", self.savedPredictons[-1].shape)
        # print("---------------")
        # print("weights shape", self.weights.shape)

    def backward(self):

        # calculate error at the output layer
        if self.activationFunction == "Sigmoid":
            outputLayerActivation = self.sigmoid(self.savedPredictons[-1])
        else:
            outputLayerActivation = self.tanh(self.savedPredictons[-1])
        output_e = self.Y_train - outputLayerActivation
        output_E = np.power(output_e, 2) * 0.5  # el mafrod vector wala rqm

        # calculate output gradient:
        # 1)
        if self.activationFunction == "Sigmoid":
            outputLocalGradient = output_e * self.sigmoidPrime(outputLayerActivation)
        else:
            outputLocalGradient = output_e * self.tanhPrime(outputLayerActivation)

        # 2)
        # partial E / partial W_kj
        self.outputLayerGradient = outputLocalGradient * self.savedPredictons[-1]
        self.gradients.append(outputLocalGradient)
        index = 0

        #####
        # calculate hidden layers gradient:
        for i in range(self.numOfHiddenLayers - 1, -1, -1):
            if self.activationFunction == "Sigmoid":
                hiddenLayerGradient = self.sigmoidPrime(self.savedPredictons[i]) * np.dot(self.gradients[index], self.weights[i + 1])
            else:
                hiddenLayerGradient = self.tanhPrime(self.savedPredictons[i]) * np.dot(self.gradients[index], self.weights[i + 1])

            # print(" self.sigmoidPrime(self.savedPredictons[i])", type(self.sigmoidPrime(self.savedPredictons[i])))
            # print("np.dot(self.gradients[index]", type((self.gradients[index])))
            # print("self.weights[i + 1]", type(self.weights[i + 1]))
            self.gradients.append(hiddenLayerGradient)
            index += 1



    def secondForward(self):

        self.gradients.reverse()
        print("len", len(self.gradients))
        for i in range(len(self.gradients)):
            print("..................")
            print("weights", self.weights[i].shape)
            print("gr T", self.gradients[i].T.shape)
            if i == 0:
                print("np.dot(self.gradients[i].T, self.train_vector.T)", np.dot(self.gradients[i].T, self.train_vector.T).shape)
                self.weights[i] = self.weights[i] + self.alpha * np.dot(self.gradients[i].T, self.train_vector.T)
                # self.bias[i] = self.bias[i] + self.alpha * np.dot(self.gradients[i].T, self.train_vector.T)
            else:
                if self.activationFunction == "Sigmoid":
                    Zj = self.sigmoidPrime(self.savedPredictons[i])
                else:
                    Zj = self.tanhPrime(self.savedPredictons[i])


                print("z.t", Zj.T.shape)
                self.weights[i] = self.weights[i] + self.alpha * np.dot(self.gradients[i].T, Zj.T)
                # self.bias[i] = self.bias[i] + self.alpha * np.dot(self.gradients[i].T, Zj.T)

    def training(self):
        for i in range(self.epochs):
            self.firstForward()
            self.backward()
            self.secondForward()

    def testing(self):
        correct = 0
        print("Ytest", self.Y_test)
        printYhat = []

        for i in range(self.test_vector.shape[1]):
            if self.use_bias_bool == 1:
                prediction = np.dot(self.weights, self.test_vector[:, i]) + self.bias
            else:
                prediction = np.dot(self.weights, self.test_vector[:, i])

            yHat = self.signum(prediction)
            self.test_predictions.append(yHat)
            error = self.Y_test[i] - yHat
            printYhat.append(yHat)

            if error == 0:
                correct += 1
        print("Yhat", printYhat)

        self.accuracy = (correct / self.test_vector.shape[1]) * 100

        # if prediction[i] != the expected class --> prediction[i] = the other class
        # if condition is true,if the condition is false
        self.test_predictions[0:20] = np.where(self.test_predictions[0:20] == self.Y_test[0], self.Y_test[0],
                                               self.Y_test[-1])
        self.test_predictions[20:] = np.where(self.test_predictions[20:] == self.Y_test[-1], self.Y_test[-1],
                                              self.Y_test[0])

        return self.accuracy, self.test_predictions

    def testInputData(self, testFeature1, testFeature2):
        prediction = (self.weights[0, 0] * float(testFeature1)) + (self.weights[0, 1] * float(testFeature2)) + self.bias
        yHat = self.signum(prediction)

        return yHat  # + 1

    def signum(self, prediction):
        if prediction >= 0:
            return 1
        elif prediction < 0:
            return 0
