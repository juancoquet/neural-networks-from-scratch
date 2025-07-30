import nnfs
from nnfs.datasets import spiral_data  # pyright: ignore[reportUnknownVariableType]

from layer_dense import LayerDense

nnfs.init()


def main() -> None:
    inputs, _ = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(n_inputs=2, n_neurons=3)
    dense1.forward(inputs)
    print(dense1.output[:5])


if __name__ == "__main__":
    main()
