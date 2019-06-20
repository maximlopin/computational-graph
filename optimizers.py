from abc import ABC, abstractmethod
from graph import default_graph


class Optimizer(ABC):

    def __init__(self, lr):
        self._lr = lr

    @abstractmethod
    def optimize(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr):
        Optimizer.__init__(self, lr)

    def optimize(self):

        for node in default_graph.parameters:
            node.value -= (node._cumulative_consumers_grad * self._lr)


class Momentum(Optimizer):
    def __init__(self, lr):
        Optimizer.__init__(self, lr)
        self._v = 0.0
        self._mu = 0.99

    def optimize(self, error):
        error.backward()

        for param in default_graph.parameters:
            self._v = (self._mu * self._v) - (param.grad * self._lr)
            param.value += self._v
