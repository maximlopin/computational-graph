import time
from tools import myprofiler

class Graph:
    def __init__(self):
        self.gates = []
        self.parameters = []
        self.placeholders = []

    def execute(self):
        for gate in self.gates:
            gate.forward()

    def compute_grads(self):
        for gate in reversed(self.gates):
            gate.backward()

    def nullify_grads(self):
        for node in self.parameters:
            node._local_grads = None
            node._cumulative_consumers_grad = 0.0

        for node in self.gates:
            node._local_grads = None
            node._cumulative_consumers_grad = 0.0

        for node in self.placeholders:
            node._local_grads = None
            node._cumulative_consumers_grad = 0.0

default_graph = Graph()
