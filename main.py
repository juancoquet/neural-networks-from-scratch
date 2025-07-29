import numpy as np
from rich import print


def main() -> None:
    inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]
    weights = [
        [0.2, 0.8, -0.5, 1.0],  # weights for neuron 1
        [0.5, -0.91, 0.26, -0.5],  # weights for neuron 2
        [-0.26, -0.27, 0.17, 0.87],  # weights for neuron 3
    ]
    biases = [2, 3, 0.5]  # one bias for each neuron
    layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

    weights2 = [  # this layer has only three inputs, so each neuron has three weights
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13],
    ]
    biases2 = [-1, 2, -0.5]
    layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
    print(layer2_outputs)

    # the above represents 4 inputs into 3 neurons, into 3 neurons again


if __name__ == "__main__":
    main()
