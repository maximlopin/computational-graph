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

    def topologically_sorted(node):

        result = []

        def expand(node):
            for n in node.input_nodes:
                expand(n)
                result.append(n)
        expand(node)

        return result

default_graph = Graph()
