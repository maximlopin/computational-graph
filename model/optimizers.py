from abc import ABC, abstractmethod
from graph.graph import default_graph


class Optimizer(ABC):

    def __init__(self, lr, fn):
        self._lr = lr
        self.fn = fn

    @abstractmethod
    def optimize(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr, fn):
        Optimizer.__init__(self, lr, fn)

    def optimize(self):

        for parameter in default_graph.compute_grads_of(self.fn):
            dx = (parameter._cumulative_consumers_grad * self._lr)
            parameter.value -= dx
