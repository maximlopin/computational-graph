import numpy as np

from graph.gates import Sigmoid, Relu
from model.model import ClassificationModel
from model.layers import Layer
from model.loss_functions import MSE
from model.optimizers import SGD


# model = ClassificationModel([
#     Layer(28*28, 10, Sigmoid)
# ], MSE, SGD, lr=0.01)


model = ClassificationModel([
    Layer(28*28, 10, Relu)
], MSE, SGD, lr=0.0001)


import mnist

x_train, y_train = mnist.train_images(), mnist.train_labels()

# Flatten images
x_train = np.array([x.flatten() for x in x_train])

# Normalize images
x_train = x_train / 255.0

# Convert labels to vectors
y_train = np.array([np.array([int(i == y) for i in range(10)]) for y in y_train])

sample_size = 200

x_train = x_train[:sample_size]
y_train = y_train[:sample_size]

model.fit(x_train, y_train, epochs=15)
