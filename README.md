# computational-graph
Computational graph + neural network on top of it. Made for educational purposes

### What is it?
Computational graph lets you define a multivariable function (which can be a complex neural network), that you can forward-propagate to get the outputs and then backward-propagete to get the derivatives (with respect to a particular node of the graph).
Each node of the graph is a constant, variable or operation, has a value and a gradient.

### Made for educational purposes
This computational graph implementation doesn't take advantage of efficient matrix operations (a node represents a single value rather than a vector/matrix), so it's very slow (especially for deep neural networks), whereas fast computational graphs like tensorflow use multidimentional varialbes and take advantage of tools like CUDA (parallel computing), AVX instructions on CPU.

### Examples

```computational-graph/graph_test.py```

```computational-graph/model_demo.py```
