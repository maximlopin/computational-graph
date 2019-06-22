from abc import abstractmethod
import numpy as np

from graph.gates import Gate

class LossFunction(Gate):

    def __init__(self, y_hat, y):
        Gate.__init__(self, y_hat)
        self.y_hat = y_hat
        self.y = y

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _compute_local_grads(self):
        pass

class MSE(LossFunction):

    def __init__(self, y_hat, y):
        LossFunction.__init__(self, y_hat, y)

    def forward(self):
        numerical_y_hat = np.array([float(t) for t in self.y_hat])
        numerical_y = np.array([float(t) for t in self.y])
        self.difference = (numerical_y_hat - numerical_y)
        self.value = np.sum(self.difference**2) / self.difference.size

    def _compute_local_grads(self):
        self._local_grads = 2 * self.difference

    def __call__(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y

class CrossEntropy(LossFunction):
    def __init__(self, y_hat, y):
        Gate.__init__(self, y_hat, y)

    def _compute_local_grads(self):

        if_label_1_dlog = -1.0 / (self.y_hat_floats)
        if_label_0_dlog = 1.0 / (1 - self.y_hat_floats)

        a = (self.y_floats == 1.0) * if_label_1_dlog
        b = (self.y_floats == 0.0) * if_label_0_dlog

        self._local_grads = (a + b)

    def forward(self):

        self.y_hat_floats = np.array([float(x) for x in self.y_hat])
        self.y_floats = np.array([float(x) for x in self.y])

        if_label_1_log = -1.0 * np.log(self.y_hat_floats)
        if_label_0_log = -1.0 * np.log(1.0 - self.y_hat_floats)

        a = (self.y_floats == 1.0) * if_label_1_log
        b = (self.y_floats == 0.0) * if_label_0_log

        self.value = (a + b)
