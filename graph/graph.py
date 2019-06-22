from .gates import Gate
from .nodes import Parameter

class Graph:
    def __init__(self):
        self._topologically_sorted_cached = [None]

    def topologically_sorted(self, node):

        if self._topologically_sorted_cached[-1] == node:
            return self._topologically_sorted_cached

        result = []

        def expand(node):
            if isinstance(node, Gate):
                for n in node._input_nodes:
                    expand(n)

            if node not in result:
                result.append(node)


        expand(node)

        self._topologically_sorted_cached = result

        return result

    def nullify_grads(self):
        for node in self._topologically_sorted_cached:
            node._cumulative_consumers_grad = 0.0

    def compute(self, node):

        for node in self.topologically_sorted(node):
            if isinstance(node, Gate):
                node.forward()

    def compute_grads_of(self, node):
        """
        Computes grads of each node, that contributes to this node
        Yields each parameter of current function/node
        """

        for n in reversed(self.topologically_sorted(node)):
            if isinstance(n, Gate):
                n.backward()
            elif isinstance(n, Parameter):
                yield n

default_graph = Graph()
