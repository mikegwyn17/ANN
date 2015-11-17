from __future__ import print_function
import numpy as np  # used for matrix multiplication and math operations
import pdb

# A class representing a multi-layered feed-forward back-propagation artificial neural network
class NeuralNet(object):
    # initialize the Hyperparameters
    def __init__(self, hidden_layers=[32]):
        restore_these_settings = np.geterr()

        temp_settings = restore_these_settings.copy()
        temp_settings["over"] = "ignore"
        temp_settings["under"] = "ignore"

        np.seterr(**temp_settings)
        np.seterr(**restore_these_settings)

        self.input_nodes = 65  # number of input nodes + 1 for the bias
        self.hidden_layers = hidden_layers  # list containing the number of hidden nodes and number of nodes per layer
        self.output_nodes = 10  # number of output nodes
        self.alpha = 1.0  # scalar used when back propagating the errors

        # initialize the weights which are is a list of nxm 2d arrays where each weight corresponds to a layer
        self.weights = [
            np.random.random_sample((self.input_nodes, self.hidden_layers[0])) * .09 + .01
        ]

        for i in range(len(hidden_layers) - 1):
            self.weights.append(
                np.random.random_sample((self.hidden_layers[i], self.hidden_layers[i + 1])) * .09 + .01)

        self.weights.append(
            np.random.random_sample((self.hidden_layers[-1], self.output_nodes)) * .09 + .01)

    # sigmoid function used to activate each of the layers
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # feed the inputs through the neural network
    def feed_forward(self, data):
        inputs = np.array(data) / 16.0  # normalize the data by dividing it by the max value
        layer = np.dot(inputs, self.hidden_layers[0])
        layer = self.sigmoid(layer)
        layers = [layer]

        for i in range(len(self.weights)):
            layer = np.dot(layer, self.weights[i])
            layer = self.sigmoid(layer)
            layers.append(layer)

        # for i in layers:
        #     print(i.shape)

        return layers

    # Derivative of the sigmoid function
    def sigmoid_prime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    # method used to train the neural network
    def train(self, inputs, answers):
        layers = self.feed_forward(inputs)
        self.deltas = self.delta(layers[-1], answers)
        self.back_propagate(layers)
        layers = self.feed_forward(inputs)
        print (layers)
        print (self.accuracy(layers, answers))

    # Calculate the delta for each of the final values,
    # and use it to back propagate the error values through the weights
    def delta(self, yHat, answers):
        list_delta = []
        temp = []
        for y in range(len(yHat)):
            for index in range(10):
                if answers[y] == index:
                    temp.append(self.sigmoid_prime(yHat[y][index]) * (answers[y] - (yHat[y][index])))
                else:
                    temp.append(self.sigmoid_prime(yHat[y][index]) * 0 - yHat[y][index])
            list_delta.append(np.array(temp))
            temp = []
            delta = np.array(list_delta)
            deltas = [delta]

        return deltas

    def back_propagate(self, layers):
        # calculate the deltas, currently hardcoded
        self.deltas.append((np.dot(self.deltas[-1], self.weights[-1].T)) * self.sigmoid_prime(layers[-2]))
        self.deltas.append((np.dot(self.deltas[-1], self.weights[-2].T)) * self.sigmoid_prime(layers[-3]))

        # calculate the weights, currently hardcoded
        self.weights[-1] = self.weights[-1] + (np.dot(layers[-2].T, self.deltas[0])) * self.alpha
        self.weights[-2] = self.weights[-2] + (np.dot(layers[-3].T, self.deltas[1])) * self.alpha

        for i in self.weights:
            print (i)

    def accuracy (self, output, answers):
        count = 0
        examples = output.shape[0]
        for i in range(examples):
            highest = np.amax(output[i])
            if highest > 0 and highest == float(output[i][answers[i]]):
                count += 1
        return count / examples
