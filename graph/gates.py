from abc import abstractmethod
import numpy as np

from .nodes import Node

class Gate(Node):
    def __init__(self, input_nodes):
        Node.__init__(self)

        self._input_nodes = input_nodes
        for node in self._input_nodes:
            node._has_consumers = True

        self._local_grads = None

    @property
    def input_floats(self):
        return [float(x) for x in self._input_nodes]

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _compute_local_grads(self):
        pass

    def backward(self):
        self._compute_local_grads()
        self.distribute_grads()

    def distribute_grads(self):
        """
        Distributes grads to input nodes (dself/dx_i * upper_level_grad)
        """

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
        self.input_floats_cached = self.input_floats
        self.value = np.prod(self.input_floats_cached)

    def _compute_local_grads(self):
        num = self.input_floats_cached
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
        self.value = sum(float(x) for x in self._input_nodes)

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
        self.value = 1.0 / (1.0 + np.exp(-1.0 * float(self.__x)))

    def _compute_local_grads(self):
        self._local_grads = [self.value*(1-self.value)]

class Logp(Gate):
    """
    Forward computes softmax(x_vector)
    Gradient is computed of log(softmax)
    """

    def __init__(self, x_vector):
        Gate.__init__(self, x_vector)

        self.x_vector = x_vector

    def _compute_local_grads(self):
        self._local_grads = (1.0 - self.value)

    def forward(self):
        numerical_x = np.array([float(x) for x in self.x_vector])
        exp_x = np.exp(numerical_x)
        self.value = (exp_x - np.max(numerical_x)) / exp_x.sum()
