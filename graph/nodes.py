from abc import ABC, abstractmethod
import numpy as np
from .graph import default_graph

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
