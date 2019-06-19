from graph_nodes import *

a = Parameter(3)
b = Parameter(4)
c = Parameter(5)
product = MulGate([a, b, c])
y = AddGate([product, c])

default_graph.execute()
default_graph.compute_grads()

print(default_graph.gates)
print(default_graph.parameters)
