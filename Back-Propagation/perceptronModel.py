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
        self.bias = None
        self.alpha = alpha
        self.test_predictions = []
        self.accuracy = 0
        self.savedActivations = []
        self.numOfHiddenLayers = numOfHiddenLayers  # int
        self.numOfNeurons = numOfNeurons  # list shayla 3dd el nuerons fe kol layer 3la 7sb el index
        self.activationFunction = activationFunction

    def sigmoid(self, prediction):
        return (1 / 1 + np.exp(-prediction))

    def sigmoidPrime(self, prediction):
        dx = self.sigmoid(prediction)
        return dx(1 - dx)

    def tanh(self, prediction):
        return ((np.exp(prediction) - np.exp(-prediction)) / (np.exp(prediction) - np.exp(-prediction)))

    def tanhPrime(self, prediction):
        dx = self.tanh(prediction)
        return (1 - pow(dx, 2))

    def firstForward(self):

        # initialize random weights matrices
        for i in range(self.numOfHiddenLayers + 1):  # +1: hidden layers + output layer
            if i == 0:
                self.weights.append(np.random.rand(self.numOfNeurons[i], len(self.train_vector)))

            else:
                self.weights.append(np.random.rand(self.numOfNeurons[i], self.numOfNeurons[i - 1]))

        self.weights = np.asarray(self.weights)

        # Algo:(

        for HL in range(self.numOfHiddenLayers + 1):
            if self.use_bias_bool == 1:
                if HL == 0:
                    prediction = np.dot(self.weights[HL], self.train_vector) + self.bias
                else:
                    prediction = np.dot(self.weights[HL], self.savedActivations[HL - 1]) + self.bias

            else:
                if HL == 0:
                    prediction = np.dot(self.weights[HL], self.train_vector)
                else:
                    prediction = np.dot(self.weights[HL], self.savedActivations[HL - 1])

            if self.activationFunction == "Sigmoid":
                activation = self.sigmoid(prediction)
            else:
                activation = self.tanh(prediction)

            self.savedActivations.append(activation)

        self.savedActivations = np.asarray(self.savedActivations)
        print(self.savedActivations.shape)

    ############################################################
    def training(self, MSEthreshold):
        self.weights = np.random.rand(1, 2)
        self.bias = np.random.rand()

        while True:
            MSE = 0
            for j in range(self.train_vector.shape[1]):
                if self.use_bias_bool == 1:
                    prediction = np.dot(self.weights, self.train_vector[:, j]) + self.bias
                else:
                    prediction = np.dot(self.weights, self.train_vector[:, j])

                error = self.Y_train[j] - prediction
                self.weights = self.weights + (error * self.alpha * self.train_vector[:, j]).T
                if self.use_bias_bool == 1:
                    self.bias = self.bias + (error * self.alpha)

            # calc MSE after each epoch
            for j in range(self.train_vector.shape[1]):
                if self.use_bias_bool == 1:
                    prediction = np.dot(self.weights, self.train_vector[:, j]) + self.bias
                else:
                    prediction = np.dot(self.weights, self.train_vector[:, j])

                error = np.power((self.Y_train[j] - prediction), 2)
                MSE += error * 0.5

            MSE /= self.train_vector.shape[1]
            self.epochs -= 1
            if MSE <= MSEthreshold or self.epochs == 0:
                break
        print("Epochs", self.epochs)
        print("MSE", MSE)
        print("bias", self.bias)
        print("weights", self.weights)
        return self.weights, self.bias

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
