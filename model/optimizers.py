from abc import ABC, abstractmethod
import numpy as np

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

class Momentum(Optimizer):
    def __init__(self, lr, fn, mu=0.80):
        Optimizer.__init__(self, lr, fn)
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

class AdaGrad(Optimizer):
    def __init__(self, lr, fn, eps=1e-7):
        Optimizer.__init__(self, lr, fn)

        self._eps = eps

        for node in default_graph.topologically_sorted(self._fn):
            if isinstance(node, Parameter):
                setattr(node, 'dx_cache', 0.0)

    def __del__(self):
        for node in default_graph.topologically_sorted(self._fn):
            if isinstance(node, Parameter):
                setattr(node, 'dx_cache', 0.0)

    def optimize(self):
        for parameter in default_graph.compute_grads_of(self._fn):
            dx = parameter._cumulative_consumers_grad
            parameter.dx_cache += np.abs(dx)
            parameter.value -= self._lr * dx / (parameter.dx_cache + self._eps)

class RMSprop(AdaGrad):
    def __init__(self, lr, fn, eps=1e-7, decay_rate=0.90):
        AdaGrad.__init__(self, lr, fn, eps)

        self._decay_rate = decay_rate

    def optimize(self):
        for parameter in default_graph.compute_grads_of(self._fn):

            dx = parameter._cumulative_consumers_grad

            parameter.dx_cache = parameter.dx_cache * self._decay_rate + (1 - self._decay_rate) * (dx**2)

            parameter.value -= self._lr * dx / (np.sqrt(parameter.dx_cache) + self._eps)

class Adam(Optimizer):
    def __init__(self, lr, fn, beta1=0.90, beta2=0.999, eps=1e-8):
        Optimizer.__init__(self, lr, fn)

        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

        for node in default_graph.topologically_sorted(self._fn):
            if isinstance(node, Parameter):
                setattr(node, 'm', 0.0)
                setattr(node, 'v', 0.0)

    def __del__(self):
        for node in default_graph.topologically_sorted(self._fn):
            if isinstance(node, Parameter):
                delattr(node, 'm')
                delattr(node, 'v')

    def optimize(self):
        for parameter in default_graph.compute_grads_of(self._fn):

            dx = parameter._cumulative_consumers_grad

            m = self._beta1 * parameter.m + (1.0 - self._beta1) * dx
            parameter.m = m

            v = self._beta2 * parameter.v + (1.0 - self._beta2) * (dx**2)
            parameter.v = v

            parameter.value -= self._lr * m / (np.sqrt(v) + self._eps)
