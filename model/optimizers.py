from abc import ABC, abstractmethod
from graph.graph import default_graph
from graph.nodes import Parameter

class Optimizer(ABC):

    def __init__(self, lr, fn):
        self._lr = lr
        self._fn = fn

    @abstractmethod
    def optimize(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr, fn):
        Optimizer.__init__(self, lr, fn)

    def optimize(self):
        for parameter in default_graph.compute_grads_of(self._fn):
            dx = (parameter._cumulative_consumers_grad * self._lr)
            parameter.value -= dx

class Momentum(SGD):
    def __init__(self, lr, fn, mu=0.80):
        SGD.__init__(self, lr, fn)
        self._mu = mu

        # Assign a "velocity" attribute for each parameter
        for node in default_graph.topologically_sorted(self._fn):
            if isinstance(node, Parameter):
                setattr(node, 'v', 0.0)

    def __del__(self):

        # Remove "velocity" attributes assigned to parameters
        for node in default_graph.topologically_sorted(self._fn):
            if isinstance(node, Parameter):
                delattr(node, 'v')

    def optimize(self):
        for parameter in default_graph.compute_grads_of(self._fn):
            dx = parameter._cumulative_consumers_grad
            parameter.v = (parameter.v * self._mu) - (dx * self._lr)

            parameter.value += parameter.v

class NAG(Momentum):
    def __init__(self, lr, fn, mu=0.80):
        Momentum.__init__(self, lr, fn, mu)

    def optimize(self):
        for parameter in default_graph.compute_grads_of(self._fn):

            dx = parameter._cumulative_consumers_grad

            v_prev = parameter.v
            parameter.v = self._mu * parameter.v - self._lr * dx

            parameter.value += -1.0 * self._mu * v_prev + (1.0 + self._mu) * parameter.v
