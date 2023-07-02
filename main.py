from neural_network import Dense, Network, Activation, Layer, Output
import numpy as np


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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    net = Network()
    net.add(Dense(2))
    net.add(Activation(relu, d_relu))
    net.add(Dense(5))
    net.add(Activation(relu, d_relu))
    net.add(Dense(4))
    net.add(Output(1, relu, d_relu))
    net.setup_loss(loss, d_loss)
    net.fit(x_train, y_train, epochs=10000, lr=0.005)
    out1 = net.predict([[0, 1]])
    print(out1)
    out0 = net.predict([[1, 1]])
    print(out0)
