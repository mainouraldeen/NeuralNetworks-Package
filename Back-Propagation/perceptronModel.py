import numpy as np


class perceptronModel:
    def __init__(self, train_vector, test_vector, Y_train, Y_test, epochs, use_bias_bool, alpha):
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

    def signum(self, prediction):
        if prediction >= 0:
            return 1
        elif prediction < 0:
            return 0

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
