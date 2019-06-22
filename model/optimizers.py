from abc import ABC, abstractmethod
from graph.graph import default_graph


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
