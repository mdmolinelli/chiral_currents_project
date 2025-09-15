import numpy as np
import sys
import psutil
import os

import matplotlib.pyplot as plt

import faulthandler
faulthandler.enable(all_threads=True)
# also write to a file for later inspection
import sys
f = open('faulthandler.log','w')
faulthandler.enable(file=f, all_threads=True)

sys.path.append(r"C:\Users\mattm\OneDrive\Desktop\Research\Projects\Triangle Lattice\Jupyter Notebooks\8Q_Triangle_Lattice_v1")
from current_measurements.src.src_current_measurement_simulations import CurrentMeasurementSimulation

def add_phase(simulation, pair_1, pair_2, angles, print_logs=False):
    number_operators = simulation.number_operators
    psi0 = simulation.psi0
    z_1 = (number_operators[pair_1[0]] - number_operators[pair_1[1]])
    z_2 = (number_operators[pair_2[0]] - number_operators[pair_2[1]])
    unitary = (1j*(angles[0]*z_1/2 + angles[1]*z_2/2)).expm()
    psi0 = simulation.psi0
    if psi0.isoper:
        if print_logs:
            print('density matrix')
        simulation.psi0 = unitary @ psi0 * unitary.dag()
    elif psi0.isket:
        if print_logs:
            print('state vector')
        simulation.psi0 = unitary @ psi0

def add_x_rotation(simulation, pair_1, pair_2, angles, print_logs=False):
    annihilation_operators = simulation.annihilation_operators
    psi0 = simulation.psi0
    a1 = annihilation_operators[pair_1[0]]
    a2 = annihilation_operators[pair_1[1]]
    a3 = annihilation_operators[pair_2[0]]
    a4 = annihilation_operators[pair_2[1]]
    x_1 = (a1.dag()*a2 + a2.dag()*a1)
    x_2 = (a3.dag()*a4 + a4.dag()*a3)
    unitary = (1j*(angles[0]*x_1/2 + angles[1]*x_2/2)).expm()
    psi0 = simulation.psi0
    if psi0.isoper:
        if print_logs:
            print('density matrix')
        simulation.psi0 = unitary @ psi0 * unitary.dag()
    elif psi0.isket:
        if print_logs:
            print('state vector')
        simulation.psi0 = unitary @ psi0

num_levels = 2
num_qubits = 4
num_particles = 2

J = 6 * 2 * np.pi
J_parallel = -6*2*np.pi

U = -180 * 2 * np.pi

measurement_detunings = np.array([300, 300, -200, -200])*2*np.pi

times = np.linspace(0, 0.2, 101)

readout_pair_1 = [0, 1]
readout_pair_2 = [2, 3]

initial_detunings = np.array([0, 0, 0, 0])*2*np.pi

psi0 = -1

current_measurement_simulation = CurrentMeasurementSimulation(num_levels, num_qubits, num_particles, J, J_parallel, U, times, readout_pair_1, readout_pair_2,
                                                              initial_detunings=None, psi0=psi0, print_logs=True)


# psi0 = current_measurement_simulation.psi0

# H = current_measurement_simulation.measurement_Hamiltonian

# number_operators = current_measurement_simulation.number_operators
# annihilation_operators = current_measurement_simulation.annihilation_operators


# phase_angles = [0, 0]
# psi0 = add_phase(psi0, number_operators, readout_pair_1, readout_pair_2, phase_angles)

# x_angles = [0, 0]
# psi0 = add_x_rotation(psi0, annihilation_operators, readout_pair_1, readout_pair_2, x_angles)