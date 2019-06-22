from abc import ABC, abstractmethod
import numpy as np

from graph.graph import default_graph
from graph.nodes import Placeholder

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

            # i.e. layer[l].input = Layer[l-1].output
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

    def __init__(self, layers, cost_f, optimizer, lr=0.01):
        Model.__init__(self, layers)

        self.cost_f = cost_f(
            y_hat=self.layers[-1].a,
            y=[None for _ in range(self.layers[-1].out_s)]
        )

        self.optimizer = optimizer(lr)

    def fit(self, x_train, y_train, epochs=15):
        total = 0
        correct = 0
        for _ in range(epochs):
            for x, y in zip(x_train, y_train):

                total += 1


                self.cost_f.y = y

                self.forward(x)

                correct += (np.argmax([float(a) for a in self.layers[-1].a]) == np.argmax(y))

                default_graph.nullify_grads()
                default_graph.compute_grads()
                self.optimizer.optimize()

                print('Epoch {}/{}. Error: {:3f}. Accuracy: {:3f}'.format(
                    _+1,
                    epochs,
                    self.cost_f.value,
                    correct/(total+1)
                ), end='\r')
