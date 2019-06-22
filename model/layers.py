import numpy as np
from graph.nodes import Placeholder, Parameter
from graph.gates import AddGate, MulGateAB

class Layer:

    def __init__(self, in_s, out_s, activation):
        self.activation = activation # Activation function
        self.in_s = in_s # Input size
        self.out_s = out_s # Output size
        self.x = None # Layer input
        self.W = None # Weights
        self.b = None # Bias
        self.z = None # Weighted sum + bias
        self.a = None # Activations

    def build(self, x):
        """
        Sets input, initializes parameters and computations
        """

        self.x = x

        # Xavier weight initialization is used
        self.W = np.array([[Parameter(np.random.rand() / np.sqrt(self.in_s))
                                        for j in range(self.in_s)]
                                            for i in range(self.out_s)])

        self.b = np.array([Parameter(0.1) for _ in range(self.out_s)])

        self.z = np.array([AddGate([AddGate([MulGateAB(self.W[i][j], self.x[j])
                                                    for j in range(self.in_s)]), self.b[i]])
                                                        for i in range(self.out_s)])
        self.a = np.array([self.activation(z) for z in self.z])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.out_s)
