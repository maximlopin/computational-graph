from abc import ABC, abstractmethod
import numpy as np
from graph import *


class Node(ABC):
    def __init__(self):
        self._has_consumers = False
        self._cumulative_consumers_grad = 0.0

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.value,
            self._cumulative_consumers_grad
        )

    def __float__(self):
        return float(self.value)

class Placeholder(Node):
    def __init__(self, value=None):
        Node.__init__(self)
        default_graph.placeholders.append(self)
        self.value = value

class Parameter(Node):
    def __init__(self, value=None):
        Node.__init__(self)
        default_graph.parameters.append(self)
        self.value = value

class Gate(Node):
    def __init__(self, input_nodes):
        Node.__init__(self)
        default_graph.gates.append(self)

        self._input_nodes = input_nodes
        for node in self._input_nodes:
            node._has_consumers = True

        self._local_grads = None
        self._cumulative_consumers_grad = 0.0

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _compute_local_grads(self):
        pass

    def backward(self):
        self._compute_local_grads()

        if self._has_consumers:
            dz = self._cumulative_consumers_grad
        else:
            dz = 1.0

        for node, grad in zip(self._input_nodes, self._local_grads):
            if isinstance(node, Gate) or isinstance(node, Parameter):
                node._cumulative_consumers_grad += (dz * grad)

class MulGate(Gate):

    def __init__(self, input_nodes):
        Gate.__init__(self, input_nodes)
        self.value = None

    def forward(self):
        self.value = np.prod([float(x) for x in self._input_nodes])

    def _compute_local_grads(self):
        num = [float(x) for x in self._input_nodes]
        self._local_grads = [np.prod(num[:i] + num[i + 1:]) for i in range(len(num))]

class AddGate(Gate):

    def __init__(self, input_nodes):
        Gate.__init__(self, input_nodes)
        self.value = None

    def forward(self):
        self.value = np.sum([float(x) for x in self._input_nodes])

    def _compute_local_grads(self):
        self._local_grads = np.ones(len(self._input_nodes))

class Relu(Gate):

    def __init__(self, x):
        Gate.__init__(self, [x])
        self.__x = x
        self.value = None

    def forward(self):
        self.value = np.maximum(0.0, float(self.__x))

    def _compute_local_grads(self):
        self._local_grads = [float(float(self.__x) > 0.0)]

class MSE(Gate):

    def __init__(self, y_hat=[], y=[]):
        Gate.__init__(self, y_hat)
        self._y_hat = y_hat
        self._y = y

    def forward(self):
        self.value = sum([(float(y_hat_i) - float(y_i))**2 for y_hat_i, y_i in zip(self._y_hat, self._y)])

    def _compute_local_grads(self):
        self._local_grads = [2 * (float(y_hat_i) - float(y_i)) for y_hat_i, y_i in zip(self._y_hat, self._y)]
