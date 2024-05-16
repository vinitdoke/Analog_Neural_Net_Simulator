import numpy as np
import matplotlib.pyplot as plt

import PySpice
from PySpice.Spice.Netlist import Circuit
import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()


class SimpleCrossbar:

    def __init__(self, input_voltage_vector, weight_matrix):

        # assert dimensions : 1 for input vector, 2 for weight matrix
        assert len(input_voltage_vector.shape) == 1
        assert len(weight_matrix.shape) == 2

        self.in_dim = len(input_voltage_vector)
        self.out_dim = len(weight_matrix[0])

        self.weight_matrix = weight_matrix
        self.input_voltage_vector = input_voltage_vector

        self.crossbar = self.create_crossbar()
        # print(self.crossbar)

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

        return self.crossbar

    def simulate(self):

        simulator = self.crossbar.simulator(temperature=25, nominal_temperature=25)
        simulator.save_currents = True

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
    pass


class StochasticCrossbar:
    pass


class MultiBitCrossbar:
    pass


def test_matrix():
    pass


if __name__ == "__main__":

    in_dim = 784
    out_dim = 16
    weight_matrix = np.random.rand(in_dim, out_dim)  # positive weights from 0 to 1
    input_voltage_vector = np.random.randn(in_dim)  # random input voltages

    # crossbar = SimpleCrossbar(input_voltage_vector, weight_matrix)

    # true matmul product
    true_output = np.dot(weight_matrix.T, input_voltage_vector)

    # crossbar product
    computed_output = SimpleCrossbar(input_voltage_vector, weight_matrix).matmul()

    print(f"True output: {true_output}")
    print(f"Computed output: {computed_output}")
    print(f"Error: {np.linalg.norm(true_output - computed_output, ord=2)}")

    # test_matrix()
