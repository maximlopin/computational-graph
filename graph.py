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

default_graph = Graph()
