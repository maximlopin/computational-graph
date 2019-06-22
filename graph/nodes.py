from abc import ABC, abstractmethod
import numpy as np

class Node(ABC):
    def __init__(self):
        self._has_consumers = False

        # Sum of partial derivatives of each consumer-gate
        # with respect to this node
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
        self.value = value

class Parameter(Node):
    def __init__(self, value=None):
        Node.__init__(self)
        self.value = value
