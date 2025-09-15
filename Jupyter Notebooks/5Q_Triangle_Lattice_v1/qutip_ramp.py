import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


def create_annihilation_operators(num_qubits, num_levels):
    annihilation_operators = []
    for i in range(num_qubits):
        operator_list = [qt.qeye(num_levels)] * num_qubits
        operator_list[i] = qt.destroy(num_levels)

        ai = qt.tensor(operator_list)
        annihilation_operators.append(ai)

    return annihilation_operators


def qubit_ramp_function(t, args):
    print(args)

    ramp_duration = args['ramp_duration']

    omega_0 = args['omega_0']
    omega_f = args['omega_f']

    if isinstance(t, (list, np.ndarray)):
        ramp = np.zeros(len(t))

        for i in range(len(ramp)):
            t_i = t[i]
            if t_i < 0:
                ramp[i] = omega_0
            elif t_i < ramp_duration:
                ramp[i] = omega_0 * (1 - t_i / ramp_duration) + omega_f * (t_i / ramp_duration)
            else:
                ramp[i] = omega_f

        return ramp
    elif isinstance(t, (float, int)):
        if t < 0:
            return omega_0
        elif t < ramp_duration:
            return omega_0 * (1 - t / ramp_duration) + omega_f * (t / ramp_duration)
        else:
            return omega_f

def create_qubit_ramp_lambda(qubit):
    return lambda t, args: qubit_ramp_function(t, args[qubit])


g = 10 * 2*np.pi # MHz * 2pi
U = -240 * 2 * np.pi # MHZ * 2pi

num_qubits = 3
num_levels = 2

annihiliation_operators = create_annihilation_operators(num_qubits, num_levels=num_levels)

H0 = 0
for a in annihiliation_operators:
    H0 += U / 2 * a.dag() * a * (a.dag() * a - 1)

for i in range(len(annihiliation_operators)):
    index_1 = i
    index_2 = (i + 1) % num_qubits

    a1 = annihiliation_operators[index_1]
    a2 = annihiliation_operators[index_2]

    H0 += g * (a1.dag() * a2 + a2.dag() * a1)

H = [H0]

qubit_to_ramp = {}
for i in range(len(annihiliation_operators)):
    qubit_ramp = create_qubit_ramp_lambda(f'{i + 1}')
    qubit_to_ramp[f'{i + 1}'] = qubit_ramp

    a = annihiliation_operators[i]
    H.append([a.dag() * a, qubit_ramp])


ramp_args = {}

ramp_args['1'] = {'ramp_duration': 1,
                'omega_0': 100*2*np.pi,
                'omega_f': 0*2*np.pi}

ramp_args['2'] = {'ramp_duration': 1,
                'omega_0': 0*2*np.pi,
                'omega_f': 0*2*np.pi}

ramp_args['3'] = {'ramp_duration': 1,
                'omega_0': 0*2*np.pi,
                'omega_f': 0*2*np.pi}


psi0 = qt.basis([num_levels]*num_qubits, [1, 0, 0])

times = np.linspace(0, 2, 1001)

e_ops = [a.dag()*a for a in annihiliation_operators]

import pdb
pdb.set_trace()
result = qt.sesolve(H, psi0, times, e_ops=e_ops, args={'1': 'h'})
expectation_values = result.expect
