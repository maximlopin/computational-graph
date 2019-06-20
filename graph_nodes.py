from abc import ABC, abstractmethod
import numpy as np
from graph import *
from tools import myprofiler

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

class MulGateAB(Gate):
    """
    Mul gate with 2 input values
    """

    def __init__(self, a, b):
        Gate.__init__(self, [a, b])
        self.value = None
        self.a = a
        self.b = b

    def forward(self):
        self.value = float(self.a) * float(self.b)

    def _compute_local_grads(self):
        self._local_grads = [float(self.b), float(self.a)]

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

class Sigmoid(Gate):

    def __init__(self, x):
        Gate.__init__(self, [x])
        self.__x = x
        self.value = None

    def forward(self):
        exp = np.exp(float(self.__x))
        self.value = exp/(1.0 + exp)

    def _compute_local_grads(self):
        self._local_grads = [self.value*(1-self.value)]

class MSE(Gate):

    def __init__(self, y_hat=[], y=[]):
        Gate.__init__(self, y_hat)
        self.y_hat = y_hat
        self.y = y

    def forward(self):
        self.difference = np.array([float(t) for t in self.y_hat]) - np.array([float(t) for t in self.y])
        self.value = np.sum(self.difference**2) / len(self.difference)

    def _compute_local_grads(self):
        self._local_grads = 2 * self.difference

    def __call__(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
