from abc import abstractmethod
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.d_loss = None

    def add(self, layer):
        self.layers.append(layer)

    def setup_loss(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss

    def predict(self, input):
        """

        :param input: [[1, 3], [2, 4]...]
        :return:
        """
        result = []
        n = len(input)
        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward(output)

            result.append(output)
        return result

    def fit(self, x_train, y_train, lr, epochs):
        n = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(n):
                # forward
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                err += self.loss(y_train[j], output)
                error = self.d_loss(y_train[j], output)
                # backward
                for layer in reversed(self.layers):
                    error = layer.backward(error, lr)

            err = err / n
            count_epoch = i + 1
            if count_epoch == 1 or count_epoch % 1000 == 0 or count_epoch == epochs:
                print('epoch: %d/%d err =:%f' % (count_epoch, epochs, err))


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None
        self.weights = None
        self.bias = None

    @abstractmethod
    def input(self):
        return self.input

    @abstractmethod
    def output(self):
        return self.output

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, prev_error, lr):
        raise NotImplementedError


class Activation(Layer):
    def __init__(self, input_shape, output_shape, activation, d_activation):
        """

        :param input_shape: (1,3)
        :param output_shape: (x,x)
        :param activation: function
        :param d_activation: function
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, prev_error, lr):
        return self.d_activation(self.input) * prev_error


class Dense(Layer):
    def __init__(self, input_shape, output_shape):
        """

        :param input_shape: (1,3)
        :param output_shape: (1,4)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1])
        self.bias = np.random.rand(1, output_shape[1]) - 0.5

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, prev_error, lr):
        current_error = np.dot(prev_error, self.weights.T)
        dweight = np.dot(self.input.T, prev_error)
        self.weights -= dweight * lr
        self.bias -= prev_error * lr
        return current_error


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def loss(y_true, y_pred):
    return 0.5 * (y_pred - y_true) ** 2


def d_loss(y_true, y_pred):
    return y_pred - y_true


if __name__ == '__main__':
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    net = Network()
    net.add(Dense((1, 2), (1, 3)))
    net.add(Activation((1, 3), (1, 3), relu, d_relu))
    net.add(Dense((1, 3), (1, 1)))
    net.add(Activation((1, 1), (1, 1), relu, d_relu))
    net.setup_loss(loss, d_loss)
    net.fit(x_train, y_train, epochs=10000, lr=0.005)
    out1 = net.predict([[0, 1]])
    print(out1)
    out0 = net.predict([[1, 1]])
    print(out0)

