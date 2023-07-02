from abc import abstractmethod
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.d_loss = None

    def add(self, layer):
        self.layers.append(layer)
        # connect dense layer only
        dense_layers = [layer for layer in self.layers if not isinstance(layer, Activation)]
        n = len(dense_layers)
        if n > 1:
            dense_layers[n - 2].connect(dense_layers[n - 1])

        # connect all layer
        n = len(self.layers)
        if n > 1:
            self.layers[n - 2].connect(self.layers[n - 1])

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
                print('epoch: %d/%d loss =:%f' % (count_epoch, epochs, err))


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

    @abstractmethod
    def connect(self, next_layer):
        raise NotImplementedError


class Activation(Layer):
    def __init__(self, activation, d_activation):
        """

        :param input_shape: (1,3)
        :param output_shape: (x,x)
        :param activation: function
        :param d_activation: function
        """
        self.input_shape = None
        self.output_shape = None
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, prev_error, lr):
        return self.d_activation(self.input) * prev_error

    def connect(self, next_layer):
        self.input_shape = self.units
        self.output_shape = next_layer.units
        if isinstance(next_layer, Dense):
            return
        if isinstance(next_layer, Activation):
            return


class Dense(Layer):
    def __init__(self, units):
        """

        :param input_shape: (1,3)
        :param output_shape: (1,4)
        """
        self.input_shape = None
        self.output_shape = None
        self.units = units
        # self.weights = np.random.rand(input_shape[1], output_shape[1])
        # self.bias = np.random.rand(1, output_shape[1]) - 0.5

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

    def connect(self, next_layer):
        self.input_shape = self.units
        if isinstance(next_layer, Activation) and not isinstance(next_layer, Output):
            next_layer.units = self.units
        if isinstance(next_layer, Dense) or isinstance(next_layer, Output):
            self.output_shape = next_layer.units
            self.weights = np.random.rand(self.units, next_layer.units)
            self.bias = np.random.rand(1, next_layer.units) - 0.5


class Output(Activation):
    def __init__(self, units, activation, d_activation):
        self.units = units
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, prev_error, lr):
        return self.d_activation(self.input) * prev_error

    def connect(self, next_layer):
        pass
