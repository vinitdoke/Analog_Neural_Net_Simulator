import numpy as np

import PySpice
from PySpice.Spice.Netlist import SubCircuit, Circuit, SubCircuitFactory
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary

import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()

libraries_path = find_libraries()
spice_library = SpiceLibrary(libraries_path)


class Cell_1T1R(SubCircuit):

    __nodes__ = ("source", "out", "gate")

    def __init__(self, name, r1):

        super().__init__(name, *self.__nodes__)
        self.MOSFET(
            1, "drain", "gate", "source", self.gnd, model="nmos"
        )  # cmos access transistor
        self.R(1, "drain", "out", r1)  # resistor


class Cell_1T1F(SubCircuit):
    
        __nodes__ = ("source", "out", "gate")
    
        def __init__(self, name, r1):  
            pass


if __name__ == "__main__":

    circuit = Circuit("One Transistor One Resistor")

    circuit.model("nmos", "nmos", level=2)
    circuit.subcircuit(Cell_1T1R("1T1R", 2))
    circuit.V("in", "source", circuit.gnd, 5)
    circuit.V("gate", "gate", circuit.gnd, 5)
    circuit.R("load", "out", circuit.gnd, 2)

    circuit.X("1T1R", "1T1R", "source", "out", "gate")

    # simulation

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    print(simulator)
    simulator.save_currents = True

    # sweep simulation
    analysis = simulator.dc(Vgate=slice(-5, 5, 0.01))

    analysis2 = simulator.dc(Vgate=slice(-5, 5, 0.01), Vin=slice(0, 5, 1))

    data = np.array(analysis2.out)
    print(data)

    # plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 10))
    plt.scatter(analysis2.sweep, analysis2.out)

    plt.xlabel("Vgate")
    plt.ylabel("Vout [V]")
    plt.grid()
    plt.show()
