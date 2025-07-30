import numpy as np
from numpy._typing import NDArray


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: NDArray[np.float64]) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases
