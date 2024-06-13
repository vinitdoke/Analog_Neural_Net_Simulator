import numpy as np
import matplotlib.pyplot as plt

import PySpice
from PySpice.Spice.Netlist import Circuit, SubCircuit
import PySpice.Logging.Logging as Logging

# from hardware.memorycells import *

from hardware.memorycells import Cell_Resistor, Cell_1T1R, Cell_1T1F

# from memorycells import Cell_Resistor

logger = Logging.setup_logging()


class Crossbar:
    """
    Skeleton class for a crossbar
    """

    def __init__(
        self,
        weight_matrix: np.ndarray,
        input_voltage_vector: np.ndarray = None,
        name: str = "Crossbar",
        verbose: bool = False,
    ):

        # dimensions
        self.in_dim = weight_matrix.shape[0]
        self.out_dim = weight_matrix.shape[1]

        self.verbose = verbose
        self.weight_matrix = weight_matrix

        # pyspice circuit object
        self.crossbar = Circuit(name)

        # set input voltages if given
        self.input_set = False
        if input_voltage_vector is not None:
            self.alter_input_voltage(input_voltage_vector)
            self.input_set = True

    def create_crossbar(self):
        raise NotImplementedError

    def alter_input_voltage(self, input_voltage_vector):

        # print(input_voltage_vector.shape, self.in_dim, self.out_dim)
        assert len(input_voltage_vector) == self.in_dim

        if self.input_set:
            # alter pre-existing voltage sources
            for i in range(self.in_dim):
                self.crossbar[f"V{i}"].dc_value = input_voltage_vector[i]

        else:
            # create new voltage sources
            for i in range(self.in_dim):
                self.crossbar.V(
                    f"{i}", f"in_{i}", self.crossbar.gnd, input_voltage_vector[i]
                )
            self.input_set = True

    def simulate(
        self, temperature=25, nominal_temperature=25
    ):  # TODO Expose SPICE Sim options
        simulator = self.crossbar.simulator(
            temperature=temperature, nominal_temperature=nominal_temperature
        )
        simulator.save_currents = True

        if self.verbose:
            print(simulator)

        analysis = simulator.operating_point()

        return analysis

    def matmul(self, input_voltage_vector=None):
        """
        Default matmul
        """
        if input_voltage_vector is not None:
            self.alter_input_voltage(input_voltage_vector)

        if not self.input_set:
            raise ValueError("Input voltage not set")

        self.analysis = self.simulate()
        products = []
        for i in range(self.out_dim):
            products.append(float(self.analysis.nodes[f"out_{i}"][0]) * 1000)

        return np.array(products)


class SimpleCrossbar(Crossbar):
    """
    Simple crossbar with no circuital non-idealities
    """

    def __init__(
        self,
        weight_matrix: np.ndarray,
        input_voltage_vector: np.ndarray = None,
        verbose: bool = False,
    ):

        super().__init__(
            weight_matrix, input_voltage_vector, verbose=verbose, name="SimpleCrossbar"
        )

        # build the resistive network
        self.create_crossbar()

    def create_crossbar(self):

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


class SimpleCrossbar2(Crossbar):
    """
    Simple crossbar with inline resistances, asbtracted memory cells, grid-based crossbar creation
    """

    def __init__(
        self,
        weight_matrix: np.ndarray,
        input_voltage_vector: np.ndarray = None,
        memory_cell: SubCircuit = Cell_Resistor,
        inline_resistances: tuple = (0, 0),
        verbose: bool = False,
    ):
        super().__init__(
            weight_matrix, input_voltage_vector, verbose=verbose, name="SimpleCrossbar2"
        )

        self.memory_cell = memory_cell
        self.inline_resistances = inline_resistances

        # build the resistive network
        self.create_crossbar()

    def create_crossbar(self):

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
        else:
            raise NotImplementedError("3 terminal not implemented yet")

        # connect all OUTs to the gnd
        for j in range(self.out_dim):
            self.crossbar.R(f"{j}", f"out_{j}", self.crossbar.gnd, 0)

        if self.verbose:
            print(self.crossbar)


class DifferentialCrossbar(Crossbar):
    """
    Differential Crossbar with inline resistances, asbtracted memory cells, grid-based crossbar creation
    Requires diffential weight matrix (m, n, 2)
    """

    def __init__(
        self,
        weight_matrix: np.ndarray,
        input_voltage_vector: np.ndarray = None,
        memory_cell: SubCircuit = Cell_Resistor,
        inline_resistances: tuple = (0, 0),
        verbose: bool = False,
    ):
        super().__init__(
            weight_matrix,
            input_voltage_vector,
            verbose=verbose,
            name="DifferentialCrossbar",
        )

        # assert weight_matrix.shape[2] == 2
        self.inline_resistances = inline_resistances
        self.memory_cell = memory_cell

        self.create_crossbar()

    def create_crossbar(self):

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

        else:
            raise NotImplementedError("3 terminal not implemented yet")

        # connect all OUTs to the gnd
        for j in range(self.out_dim):
            self.crossbar.R(f"{j}_p", f"out_{j}_p", self.crossbar.gnd, 0)
            self.crossbar.R(f"{j}_n", f"out_{j}_n", self.crossbar.gnd, 0)

        if self.verbose:
            print(self.crossbar)

    def matmul(self, input_voltage_vector=None):

        if input_voltage_vector is not None:
            self.alter_input_voltage(input_voltage_vector)

        if not self.input_set:
            raise ValueError("Input voltage not set")

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


def test_matrix(multiplier=SimpleCrossbar):

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
                weight_matrix=weight_matrix,
                input_voltage_vector=input_voltage_vector,
                memory_cell=Cell_Resistor,
                inline_resistances=(0, 1e-4),
            ).matmul()

            errors[i, j] = np.linalg.norm(
                true_output - computed_output, ord=2
            ) / np.linalg.norm(true_output, ord=2)
            # errors[i, j] = np.linalg.norm(true_output - computed_output, ord=2)

    plt.imshow(errors, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.title("Error Heatmap", fontsize=25)
    plt.xlabel("Output Dimension", fontsize=20)
    plt.ylabel("Input Dimension", fontsize=20)

    plt.xticks(np.arange(len(outputs)), outputs, fontsize=15)
    plt.yticks(np.arange(len(inputs)), inputs, fontsize=15)

    plt.savefig("errorscaling3.eps", dpi = 600, bbox_inches='tight')

    plt.show()


def test_matrix2(in_line_resistances=(1e-4, 1e-4)):

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

    plt.title("Error Heatmap", fontsize=25)
    plt.xlabel("Output Dimension", fontsize=20)
    plt.ylabel("Input Dimension", fontsize=20)

    plt.xticks(np.arange(len(outputs)), outputs, fontsize=20)
    plt.yticks(np.arange(len(inputs)), inputs, fontsize=20)

    plt.savefig("errorscaling.eps", dpi = 600, bbox_inches='tight')

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

    test_matrix()
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

    # in_dim = 3
    # out_dim = 3
    # weight_matrix = np.random.randn(in_dim, out_dim, 2)  # positive weights from 0 to 1
    # input_voltage_vector = np.random.randn(in_dim)  # random input voltages

    # true_output = (
    #     np.dot(weight_matrix.T, input_voltage_vector)[0]
    #     - np.dot(weight_matrix.T, input_voltage_vector)[1]
    # )

    # crossbar = DifferentialCrossbar(
    #     input_voltage_vector,
    #     weight_matrix,
    #     Cell_Resistor,
    #     inline_resistances=(1e-3, 1e-3),
    #     verbose=True,
    # )

    # products = crossbar.matmul()

    # print(f"{products}")
    # print(f"{true_output}")

    # error = np.linalg.norm(products - true_output, ord=2) / np.linalg.norm(
    #     true_output, ord=2
    # )
    # print(f"Error: {error}")

    ### REFACTORED
    # v1_set = np.random.randn(3)
    # v2_set = np.random.randn(3)
    # matrix = np.random.randn(3, 3, 2)

    # trial = DifferentialCrossbar(matrix, v1_set, verbose=True)
    # print(trial.matmul())
    # print(np.matmul(matrix[:, :, 0].T - matrix[:, :, 1].T, v1_set))

    # print(trial.matmul(v2_set))
    # print(np.matmul(matrix[:, :, 0].T - matrix[:, :, 1].T, v2_set))

    ### REFACTORED CROSSBAR2
    # v1_set = np.random.randn(3)
    # v2_set = np.random.randn(3)
    # matrix = np.random.randn(3, 3)

    # trial = SimpleCrossbar2(
    #     weight_matrix=matrix,
    #     input_voltage_vector=v1_set,
    #     memory_cell=Cell_Resistor,
    #     inline_resistances=(1e-3, 1e-3),
    #     verbose=True,
    # )
    # print(trial.matmul())
    # print(np.matmul(matrix.T, v1_set))
