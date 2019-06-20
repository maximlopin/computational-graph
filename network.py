from abc import ABC, abstractmethod
import numpy as np

from graph import default_graph
from graph_nodes import *
from optimizers import *

class Layer:

    def __init__(self, in_s, out_s, activation):
        self.activation = activation # Activation function
        self.in_s = in_s # Input size
        self.out_s = out_s # Output size
        self.x = None # Layer input
        self.W = None # Weights
        self.b = None # Bias
        self.z = None # Weighted sum + bias
        self.a = None # Activations

    def build(self, x):
        """
        Sets input, initializes parameters and computations
        """

        self.x = x
        self.W = np.array([[Parameter((np.random.rand()*2 - 1)/(np.sqrt(self.in_s/2)))
                                        for j in range(self.in_s)]
                                            for i in range(self.out_s)])

        self.b = np.array([Parameter(0.0) for _ in range(self.out_s)])

        self.z = np.array([AddGate([AddGate([MulGateAB(self.W[i][j], self.x[j])
                                                    for j in range(self.in_s)]), self.b[i]])
                                                        for i in range(self.out_s)])
        self.a = np.array([self.activation(z) for z in self.z])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.out_s)

class Model(ABC):
    def __init__(self, layers):
        self.layers = layers
        self.build_layers()

    def build_layers(self):
        """
        Make each layer's output the input of the next layer
        """

        # First layer's input is a vector of placeholders
        x = np.array([Placeholder() for _ in range(self.layers[0].in_s)])
        for layer in self.layers:
            layer.build(x)

            # layer[l].input = Layer[l-1].output
            x = layer.a

    def set_input(self, x):

        assert(x.shape == self.layers[0].x.shape)

        for p, x in zip(self.layers[0].x, x):
            p.value = x

    def forward(self, x):
        self.set_input(x)

        default_graph.execute()

        return self.layers[-1].a

    def __repr__(self):
        s = '{}[\n'.format(self.__class__.__name__)
        for layer in self.layers:
            s += '  {}({})\n'.format(layer.__class__.__name__, layer.out_s)
        s += ']'

        return s

class ClassificationModel(Model):

    def __init__(self, layers, cost_f, optimizer):
        Model.__init__(self, layers)

        self.cost_f = cost_f(
            y_hat=self.layers[-1].a,
            y=[None for _ in range(self.layers[-1].out_s)]
        )

        self.optimizer = optimizer(lr=0.01)

    def fit(self, x_train, y_train, epochs=15):
        total = 0
        correct = 0
        for _ in range(epochs):
            for x, y in zip(x_train, y_train):

                total += 1


                self.cost_f.y = y

                self.forward(x)

                correct += (np.argmax([float(a) for a in self.layers[-1].a]) == np.argmax(y))

                # Make sure grads are all equal to zero
                default_graph.nullify_grads()

                # Compute grads
                default_graph.compute_grads()

                # Use optimizer to apply grads to parameters
                self.optimizer.optimize()

                print('Epoch {}/{}. Error: {}. Accuracy: {}'.format(_+1, epochs, self.cost_f.value, correct/(total+1)), end='\r')
