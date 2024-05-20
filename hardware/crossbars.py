import numpy as np
import matplotlib.pyplot as plt

import PySpice
from PySpice.Spice.Netlist import Circuit, SubCircuit
import PySpice.Logging.Logging as Logging

# from hardware.memorycells import *

# from hardware.memorycells import Cell_Resistor, Cell_1T1R, Cell_1T1F
from memorycells import Cell_Resistor

# from interface.IOTransfer import weights_to_differential_resistance

logger = Logging.setup_logging()


class SimpleCrossbar:
    """
    Simple crossbar with no circuital non-idealities
    """

    def __init__(self, input_voltage_vector, weight_matrix, verbose=False):

        assert len(input_voltage_vector.shape) == 1
        assert len(weight_matrix.shape) == 2
        self.verbose = verbose

        self.in_dim = len(input_voltage_vector)
        self.out_dim = len(weight_matrix[0])

        self.weight_matrix = weight_matrix
        self.input_voltage_vector = input_voltage_vector

        self.create_crossbar()

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


class SimpleCrossbar2:
    """
    Simple crossbar with inline resistances, asbtracted memory cells, grid-based crossbar creation
    """

    def __init__(
        self,
        input_voltage_vector: np.ndarray,
        weight_matrix: np.ndarray,
        memory_cell: SubCircuit = Cell_Resistor,
        inline_resistances: tuple = (0, 0),
        verbose: bool = False,
    ):

        self.verbose = verbose
        self.weight_matrix = weight_matrix
        self.input_voltage_vector = input_voltage_vector
        self.inline_resistances = inline_resistances  # (R_vertical, R_horizontal)

        self.in_dim = len(input_voltage_vector)
        self.out_dim = len(weight_matrix[0])

        self.memory_cell = memory_cell

        self.crossbar = self.create_crossbar()

    def create_crossbar(self):

        # create circuit
        self.crossbar = Circuit("Crossbar")

        # add voltage sources
        for i in range(self.in_dim):
            self.crossbar.V(
                f"{i}", f"in_{i}", self.crossbar.gnd, self.input_voltage_vector[i]
            )

        # add horizontal in_line resistor series on each voltage input line
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                if j == 0:
                    self.crossbar.R(
                        f"source_{i}_{j}",
                        f"in_{i}",
                        f"in_{i}_{j}",
                        self.inline_resistances[1],
                    )
                else:
                    self.crossbar.R(
                        f"source_{i}_{j}",
                        f"in_{i}_{j-1}",
                        f"in_{i}_{j}",
                        self.inline_resistances[1],
                    )

        # on each output line, serially add memory cells followed by a vertical in_line resistor ending with the output node
        if len(self.memory_cell.__nodes__) == 2:
            for j in range(self.out_dim):
                for i in range(self.in_dim):
                    self.crossbar.subcircuit(
                        self.memory_cell(f"{i}_{j}", 1 / self.weight_matrix[i][j])
                    )
                    if i != self.in_dim - 1:
                        self.crossbar.X(
                            f"{i}_{j}",
                            f"{i}_{j}",
                            f"in_{i}_{j}",
                            f"intermediate_{i}_{j}",
                        )
                        self.crossbar.R(
                            f"intermediate_{i}_{j}",
                            f"intermediate_{i}_{j}",
                            f"intermediate_{i+1}_{j}",
                            self.inline_resistances[0],
                        )
                    else:
                        self.crossbar.X(
                            f"{i}_{j}", f"{i}_{j}", f"in_{i}_{j}", f"out_{j}"
                        )
                        self.crossbar.R(
                            f"intermediate_{i}_{j}",
                            f"intermediate_{i}_{j}",
                            f"out_{j}",
                            self.inline_resistances[0],
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

        products = []
        for i in range(self.out_dim):
            products.append(float(self.analysis.nodes[f"out_{i}"][0]) * 1000)

        return np.array(products)


class DifferentialCrossbar:

    def __init__(
        self,
        input_voltage_vector: np.ndarray,
        weight_matrix: np.ndarray,
        memory_cell: SubCircuit = Cell_Resistor,
        inline_resistances: tuple = (0, 0),
        verbose: bool = False,
    ):

        self.verbose = verbose
        self.weight_matrix = weight_matrix
        self.input_voltage_vector = input_voltage_vector
        self.inline_resistances = inline_resistances

        assert len(input_voltage_vector) == len(weight_matrix)
        assert len(weight_matrix.shape) == 3

        self.in_dim = len(input_voltage_vector)
        self.out_dim = len(weight_matrix[0])

        self.memory_cell = memory_cell

        self.crossbar = self.create_crossbar()

    def create_crossbar(self):

        # create circuit
        self.crossbar = Circuit("Crossbar")

        # add voltage sources
        for i in range(self.in_dim):
            self.crossbar.V(
                f"{i}", f"in_{i}", self.crossbar.gnd, self.input_voltage_vector[i]
            )

        # add horizontal in_line resistor series (2 times the normal) on each voltage input line
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                if j == 0:
                    self.crossbar.R(
                        f"source_{i}_{j}_p",
                        f"in_{i}",
                        f"in_{i}_{j}_p",
                        self.inline_resistances[1],
                    )
                    self.crossbar.R(
                        f"source_{i}_{j}_n",
                        f"in_{i}_{j}_p",
                        f"in_{i}_{j}_n",
                        self.inline_resistances[1],
                    )
                else:
                    self.crossbar.R(
                        f"source_{i}_{j}_p",
                        f"in_{i}_{j-1}_n",
                        f"in_{i}_{j}_p",
                        self.inline_resistances[1],
                    )
                    self.crossbar.R(
                        f"source_{i}_{j}_n",
                        f"in_{i}_{j}_p",
                        f"in_{i}_{j}_n",
                        self.inline_resistances[1],
                    )

        # on each output line, serially add memory cells followed by a vertical in_line resistor ending with the output node
        if len(self.memory_cell.__nodes__) == 2:
            for j in range(self.out_dim):
                for i in range(self.in_dim):
                    self.crossbar.subcircuit(
                        self.memory_cell(f"{i}_{j}_p", 1 / self.weight_matrix[i][j][0])
                    )
                    self.crossbar.subcircuit(
                        self.memory_cell(f"{i}_{j}_n", 1 / self.weight_matrix[i][j][1])
                    )

                    if i != self.in_dim - 1:
                        self.crossbar.X(
                            f"{i}_{j}_p",
                            f"{i}_{j}_p",
                            f"in_{i}_{j}_p",
                            f"intermediate_{i}_{j}_p",
                        )
                        self.crossbar.X(
                            f"{i}_{j}_n",
                            f"{i}_{j}_n",
                            f"in_{i}_{j}_n",
                            f"intermediate_{i}_{j}_n",
                        )
                        self.crossbar.R(
                            f"intermediate_{i}_{j}_p",
                            f"intermediate_{i}_{j}_p",
                            f"intermediate_{i+1}_{j}_p",
                            self.inline_resistances[0],
                        )
                        self.crossbar.R(
                            f"intermediate_{i}_{j}_n",
                            f"intermediate_{i}_{j}_n",
                            f"intermediate_{i+1}_{j}_n",
                            self.inline_resistances[0],
                        )
                    else:
                        self.crossbar.X(
                            f"{i}_{j}_p", f"{i}_{j}_p", f"in_{i}_{j}_p", f"out_{j}_p"
                        )
                        self.crossbar.X(
                            f"{i}_{j}_n", f"{i}_{j}_n", f"in_{i}_{j}_n", f"out_{j}_n"
                        )
                        self.crossbar.R(
                            f"intermediate_{i}_{j}_p",
                            f"intermediate_{i}_{j}_p",
                            f"out_{j}_p",
                            self.inline_resistances[0],
                        )
                        self.crossbar.R(
                            f"intermediate_{i}_{j}_n",
                            f"intermediate_{i}_{j}_n",
                            f"out_{j}_n",
                            self.inline_resistances[0],
                        )

        # connect all OUTs to the gnd
        for j in range(self.out_dim):
            self.crossbar.R(f"{j}_p", f"out_{j}_p", self.crossbar.gnd, 0)
            self.crossbar.R(f"{j}_n", f"out_{j}_n", self.crossbar.gnd, 0)

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

        products = []
        for i in range(self.out_dim):
            products.append(
                (
                    float(self.analysis.nodes[f"out_{i}_p"][0])
                    - float(self.analysis.nodes[f"out_{i}_n"][0])
                )
                * 1000
            )

        return np.array(products)

    def visualize(self):
        pass


class MultiBitCrossbar:
    pass


def test_matrix(multiplier=SimpleCrossbar):

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


def test_matrix2(in_line_resistances=(0, 0)):

    in_dim_min = 10
    in_dim_max = 50
    out_dim_min = 10
    out_dim_max = 50
    step_size = 5

    inputs = np.arange(in_dim_min, in_dim_max, step_size)
    outputs = np.arange(out_dim_min, out_dim_max, step_size)

    errors = np.zeros((len(inputs), len(outputs)))

    for i, in_dim in enumerate(inputs):
        for j, out_dim in enumerate(outputs):

            print(f"Input Dimension: {in_dim}, Output Dimension: {out_dim}", end="\r")

            weight_matrix = np.random.rand(in_dim, out_dim)
            input_voltage_vector = np.random.randn(in_dim)

            true_output = np.dot(weight_matrix.T, input_voltage_vector)

            computed_output = SimpleCrossbar2(
                input_voltage_vector,
                weight_matrix,
                Cell_Resistor,
                inline_resistances=in_line_resistances,
            ).matmul()

            errors[i, j] = np.linalg.norm(
                true_output - computed_output, ord=2
            ) / np.linalg.norm(true_output, ord=2)

    plt.imshow(errors, cmap="hot", interpolation="nearest")
    plt.colorbar()
    # plt.xlabel("Output Dimension")
    # plt.ylabel("Input Dimension")
    # plt.xticks(np.arange(len(outputs)), outputs)
    # plt.yticks(np.arange(len(inputs)), inputs)
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

    # # test_matrix()
    # # error_growth_over_input_dim()

    # in_dim = 10
    # out_dim = 16
    # weight_matrix = np.random.randn(in_dim, out_dim)  # positive weights from 0 to 1
    # input_voltage_vector = np.random.randn(in_dim)  # random input voltages
    # # crossbar = SimpleCrossbar(input_voltage_vector, weight_matrix)

    # # true matmul product
    # true_output = np.dot(weight_matrix.T, input_voltage_vector)

    # # crossbar product
    # # dif_crossbar = DifferentialCrossbar(input_voltage_vector, weight_matrix, verbose=True)
    # computed_output = SimpleCrossbar(
    #     input_voltage_vector, weight_matrix, verbose=True
    # ).matmul()

    # print(f"True output: {true_output}")
    # print(f"Computed output: {computed_output}")
    # print(f"Error: {np.linalg.norm(true_output - computed_output, ord=2)}")

    #### New crossbar test

    # in_dim = 2
    # out_dim = 3
    # weight_matrix = np.random.randn(in_dim, out_dim)  # positive weights from 0 to 1
    # input_voltage_vector = np.random.randn(in_dim)  # random input voltages

    # crossbar = SimpleCrossbar2(
    #     input_voltage_vector,
    #     weight_matrix,
    #     Cell_Resistor,
    #     inline_resistances=(1e-3, 1e-3),
    #     verbose=True
    # )

    # print(crossbar.matmul())
    # print(f"true output = {np.dot(weight_matrix.T, input_voltage_vector)}")

    # test_matrix2((1e-3, 20*1e-3))

    ### Differential Crossbar test

    in_dim = 3
    out_dim = 3
    weight_matrix = np.random.randn(in_dim, out_dim, 2)  # positive weights from 0 to 1
    input_voltage_vector = np.random.randn(in_dim)  # random input voltages

    true_output = np.dot(weight_matrix.T, input_voltage_vector)[0] - np.dot(weight_matrix.T, input_voltage_vector)[1]


    crossbar = DifferentialCrossbar(
        input_voltage_vector,
        weight_matrix,
        Cell_Resistor,
        inline_resistances=(1e-3, 1e-3),
        verbose=True,
    )

    products = crossbar.matmul()

    print(f"{products}")
    print(f"{true_output}")

    error = np.linalg.norm(products - true_output, ord=2)/np.linalg.norm(true_output, ord=2)
    print(f"Error: {error}")
