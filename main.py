import numpy as np
from rich import print


def main() -> None:
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1.0],  # weights for neuron 1
        [0.5, -0.91, 0.26, -0.5],  # weights for neuron 2
        [-0.26, -0.27, 0.17, 0.87],  # weights for neuron 3
    ]
    biases = [2, 3, 0.5]  # one bias for each neuron
    layer_outputs = np.dot(weights, inputs) + biases
    print(layer_outputs)


if __name__ == "__main__":
    main()
