
from graph.graph import default_graph
from graph.nodes import Parameter
from graph.gates import AddGate
from graph.gates import MulGate

a = Parameter(3)
b = Parameter(4)
c = Parameter(5)
product = MulGate([a, b, c])
y = AddGate([product, c])

default_graph.compute(y)
default_graph.compute_grads_of(y)

print(a, b, c, product, y)
