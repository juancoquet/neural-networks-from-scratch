def main() -> None:
    inputs = [1.0, 2.0, 3.0, 2.5]

    weights1 = [0.2, 0.8, -0.5, 1.0]  # weights for neuron 1
    weights2 = [0.5, -0.91, 0.26, -0.5]  # weights for neuron 2
    weights3 = [-0.26, -0.27, 0.17, 0.87]  # weights for neuron 3

    bias1, bias2, bias3 = 2, 3, 0.5  # one bias for each neuron

    output1 = _calculate_output(inputs, weights1, bias1)
    output2 = _calculate_output(inputs, weights2, bias2)
    output3 = _calculate_output(inputs, weights3, bias3)

    output = [output1, output2, output3]
    print(output)


def _calculate_output(inputs: list[float], weights: list[float], bias: float) -> float:
    return sum(i * w for i, w in zip(inputs, weights, strict=True)) + bias


if __name__ == "__main__":
    main()
