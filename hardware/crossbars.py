import numpy as np
import matplotlib.pyplot as plt

import PySpice
from PySpice.Spice.Netlist import Circuit
import PySpice.Logging.Logging as Logging

from interface.IOTransfer import weights_to_differential_conductance

# from IOTransfer import weights_to_differential_conductance

logger = Logging.setup_logging()


class SimpleCrossbar:

    def __init__(self, input_voltage_vector, weight_matrix, verbose=False):

        assert len(input_voltage_vector.shape) == 1
        assert len(weight_matrix.shape) == 2
        self.verbose = verbose

        self.in_dim = len(input_voltage_vector)
        self.out_dim = len(weight_matrix[0])

        self.weight_matrix = weight_matrix
        self.input_voltage_vector = input_voltage_vector

        self.crossbar = self.create_crossbar()

    def create_crossbar(self) -> Circuit:

        self.crossbar = Circuit("Crossbar")

        # add voltage sources
        for i in range(self.in_dim):
            self.crossbar.V(
                f"{i}", f"in_{i}", self.crossbar.gnd, self.input_voltage_vector[i]
            )

        # add resistors
        for j in range(self.out_dim):
            for i in range(self.in_dim):
                self.crossbar.R(
                    f"_{i}_{j}", f"in_{i}", f"out_{j}", 1 / self.weight_matrix[i][j]
                )

        # connect all OUTs to the gnd
        for j in range(self.out_dim):
            self.crossbar.R(f"{j}", f"out_{j}", self.crossbar.gnd, 0)

        if self.verbose:
            print(self.crossbar)

        return self.crossbar

    def simulate(self):

        simulator = self.crossbar.simulator(temperature=25, nominal_temperature=25)
        simulator.save_currents = True

        if self.verbose:
            print(simulator)

        analysis = simulator.operating_point()

        return analysis

    def matmul(self):

        self.analysis = self.simulate()

        # for node in self.analysis.nodes.values():
        #     print(f"{str(node)}: {np.array(node)[0]}V")

        # for param in self.analysis.internal_parameters.values():
        #     print(f"{str(param)}: {np.array(param)[0]}A")

        products = []
        for i in range(self.out_dim):
            products.append(float(self.analysis.nodes[f"out_{i}"][0]) * 1000)

        return np.array(products)

    def visualize(self):
        pass


class DifferentialCrossbar:

    def __init__(self, input_voltage_vector, weight_matrix, verbose=False):

        assert len(input_voltage_vector.shape) == 1
        assert len(weight_matrix.shape) == 2
        self.verbose = verbose

        self.in_dim = len(input_voltage_vector)
        self.out_dim = len(weight_matrix[0])

        self.weight_matrix = weight_matrix
        self.input_voltage_vector = input_voltage_vector

        self.resistance_matrix = weights_to_differential_conductance(weight_matrix)
        self.create_crossbar()

    def create_crossbar(self) -> Circuit:

        self.crossbar = Circuit("Differential Crossbar")

        # add voltage sources
        for i in range(self.in_dim):
            self.crossbar.V(
                f"{i}", f"in_{i}", self.crossbar.gnd, self.input_voltage_vector[i]
            )

        # add resistors

    def simulate(self):
        pass

    def matmul(self):
        pass

    def visualize(self):
        pass


class StochasticCrossbar:
    pass


class MultiBitCrossbar:
    pass


def test_matrix():

    in_dim_min = 10
    in_dim_max = 100
    out_dim_min = 10
    out_dim_max = 100
    step_size = 10

    inputs = np.arange(in_dim_min, in_dim_max, step_size)
    outputs = np.arange(out_dim_min, out_dim_max, step_size)

    errors = np.zeros((len(inputs), len(outputs)))

    for i, in_dim in enumerate(inputs):
        for j, out_dim in enumerate(outputs):

            weight_matrix = np.random.rand(in_dim, out_dim)
            input_voltage_vector = np.random.randn(in_dim)
            true_output = np.dot(weight_matrix.T, input_voltage_vector)
            computed_output = SimpleCrossbar(
                input_voltage_vector, weight_matrix
            ).matmul()
            errors[i, j] = np.linalg.norm(
                true_output - computed_output, ord=2
            ) / np.linalg.norm(true_output, ord=2)
            # errors[i, j] = np.linalg.norm(true_output - computed_output, ord=2)

    plt.imshow(errors, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Output Dimension")
    plt.ylabel("Input Dimension")
    plt.xticks(np.arange(len(outputs)), outputs)
    plt.yticks(np.arange(len(inputs)), inputs)
    plt.show()


def error_growth_over_input_dim():

    in_dim_min = 10
    in_dim_max = 100
    in_dim_step = 10
    out_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    inputs = np.arange(in_dim_min, in_dim_max, in_dim_step)
    errors = np.zeros((len(inputs), len(out_dims)))

    for i, in_dim in enumerate(inputs):
        for j, out_dim in enumerate(out_dims):

            weight_matrix = np.random.rand(in_dim, out_dim)
            input_voltage_vector = np.random.randn(in_dim)
            true_output = np.dot(weight_matrix.T, input_voltage_vector)
            computed_output = SimpleCrossbar(
                input_voltage_vector, weight_matrix
            ).matmul()
            # errors[i, j] = np.linalg.norm(true_output - computed_output, ord=2)/np.linalg.norm(true_output, ord=2)
            errors[i, j] = np.linalg.norm(true_output - computed_output, ord=2)

    for j, out_dim in enumerate(out_dims):
        plt.plot(inputs, errors[:, j], label=f"Output Dimension: {out_dim}")

    plt.xlabel("Input Dimension")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # test_matrix()
    # error_growth_over_input_dim()

    in_dim = 10
    out_dim = 16
    weight_matrix = np.random.randn(in_dim, out_dim)  # positive weights from 0 to 1
    input_voltage_vector = np.random.randn(in_dim)  # random input voltages
    # crossbar = SimpleCrossbar(input_voltage_vector, weight_matrix)

    # true matmul product
    true_output = np.dot(weight_matrix.T, input_voltage_vector)

    # crossbar product
    # dif_crossbar = DifferentialCrossbar(input_voltage_vector, weight_matrix, verbose=True)
    computed_output = SimpleCrossbar(
        input_voltage_vector, weight_matrix, verbose=True
    ).matmul()

    print(f"True output: {true_output}")
    print(f"Computed output: {computed_output}")
    print(f"Error: {np.linalg.norm(true_output - computed_output, ord=2)}")
