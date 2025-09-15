from itertools import product

import numpy as np
import qutip as qt
from scipy.optimize import minimize

from src.src_current_measurement_simulations import CurrentMeasurementSimulation

def J_to_beamsplitter_time(J):
    """
    Convert J (2pi MHz) to beamsplitter time in nanoseconds.
    """
    return np.pi/4/(J) / 1e6 * 1e9

def J_to_beamsplitter_time_samples(J):
    """
    Convert J (2pi MHz) to beamsplitter time in samples.
    """
    return np.pi/4/(J) / 1e6 * 1e9 * 16 / 2.32515*2


def cost_function(data, simulation):
    """
    Calculate the cost function as the sum squared error between the measured and simulated N-terms.
    """
    return np.sum((data - simulation) ** 2)




def calculate_populations_simulation(num_levels=None, num_qubits=None, num_particles=None, J=None, J_parallel=None, U=None, times=None, 
                                     readout_pair_1=None, readout_pair_2=None, initial_detunings=None, measurement_detuning=None, 
                                     measurement_J=None, measurement_J_parallel=None, psi0=None, 
                                     T1=None, T2=None, scale_factor=1, print_logs=False, uncoupled=False, **kwargs):


    # print("Simulation parameters:")
    # print(f"num_levels: {num_levels}")
    # print(f"num_qubits: {num_qubits}")
    # print(f"num_particles: {num_particles}")
    # print(f"J: {J}")
    # print(f"J_parallel: {J_parallel}")
    # print(f"U: {U}")
    # print(f"times: {times}")
    # print(f"readout_pair_1: {readout_pair_1}")
    # print(f"readout_pair_2: {readout_pair_2}")
    # print(f"initial_detunings: {initial_detunings}")
    # print(f"measurement_detuning: {measurement_detuning}")
    # print(f"measurement_J: {measurement_J}")
    # print(f"measurement_J_parallel: {measurement_J_parallel}")
    # print(f"psi0: {psi0}")
    # print(f"T1: {T1}")
    # print(f"T2: {T2}")
    # print(f"scale_factor: {scale_factor}")
    # print(f"print_logs: {print_logs}")
    # print(f"uncoupled: {uncoupled}")
    # print(f"kwargs: {kwargs}")

    current_measurement_simulation = CurrentMeasurementSimulation(num_levels, num_qubits, num_particles, J, J_parallel, U, times, 
                                                                  readout_pair_1, readout_pair_2, initial_detunings, measurement_detuning, 
                                                                  measurement_J=measurement_J, measurement_J_parallel=measurement_J_parallel, 
                                                                  psi0=psi0, T1=T1, T2=T2, print_logs=print_logs)

    current_measurement_simulation.psi0 = modify_initial_state(current_measurement_simulation, print_logs=print_logs, **kwargs)



    current_measurement_simulation.run_simulation(uncoupled)

    populations = scale_factor * current_measurement_simulation.get_population_average()

    return populations[np.array(readout_pair_1 + readout_pair_2)]


def calculate_covariance_simulation(num_levels, num_qubits, num_particles, J, J_parallel, U, times, readout_pair_1, readout_pair_2,
                                  initial_detunings, measurement_detuning, measurement_J=None, measurement_J_parallel=None, 
                                  psi0=None, scale_factor=1, T1=None, T2=None, print_logs=False, uncoupled=False, return_simulation_object=False, **kwargs):



    current_measurement_simulation = CurrentMeasurementSimulation(num_levels, num_qubits, num_particles, J, J_parallel, U, times, 
                                                                  readout_pair_1, readout_pair_2, initial_detunings, 
                                                                  measurement_detuning, measurement_J=measurement_J, 
                                                                  measurement_J_parallel=measurement_J_parallel, 
                                                                  psi0=psi0, time_offset=0, T1=T1, T2=T2, print_logs=print_logs)
    
    current_measurement_simulation.psi0 = modify_initial_state(current_measurement_simulation, print_logs=print_logs, **kwargs)


    current_measurement_simulation.run_simulation(uncoupled)


    population_average = current_measurement_simulation.get_population_average()

    # for now, skip i,j = 0,1 and 2,3
    covariance_simulation = []
    # for i in range(num_qubits):
    #     for j in range(i+1, num_qubits):
    #         if (i == 0 and j == 1) or (i == 2 and j == 3):
    #             continue

    for i in readout_pair_1:
        for j in readout_pair_2:
            covariance_simulation.append(scale_factor*(current_measurement_simulation.get_n_term(i, j) - population_average[i,:]*population_average[j,:]))

    covariance_simulation = np.array(covariance_simulation)

    if return_simulation_object:
        return covariance_simulation, current_measurement_simulation
    else:
        return covariance_simulation

def modify_initial_state(current_measurement_simulation, print_logs=False, **kwargs):
    '''
    add_phase: Adds a phase between pairs of qubits
        pairs: pairs of qubit indices
        phases: corresponding phases to apply

    use_mixed_state: creates a mixed state from the given parameters
        mixed_basis_states: list of bitstrings representing the basis states of the mixed state
        mixed_probabilities: list of probabilities for each basis state
        pre_ramp_detuning: specifies energy detuning before the ramp, used to determine ordering of basis states
        basis_state_to_index_dict: mapping from basis states to their index within the list of states with the same particle number
    # TODO: modify this to use measurements of occurances after ramp, not before
    '''

    initial_state = current_measurement_simulation.psi0

    if print_logs:
        if initial_state.isket:
            print(f'initial state before modifying is state vector')
        elif initial_state.isoper:
            print(f'initial state before modifying is density matrix')


    if kwargs.get('use_mixed_state', False):
        mixed_basis_states = kwargs['mixed_basis_states']
        mixed_probabilities = kwargs['mixed_probabilities']
        basis_state_to_index_dict = kwargs['basis_state_to_index_dict']

        if print_logs:
            print(f'creating mixed state')
            print(f'states: {mixed_basis_states}')
            print(f'probabilities: {mixed_probabilities}')

        initial_state = create_mixed_state(current_measurement_simulation, mixed_basis_states, mixed_probabilities, 
                                           basis_state_to_index_dict, print_logs=print_logs)


    if kwargs.get('use_mixed_ramp'):
        mixed_basis_states = kwargs['mixed_basis_states']
        mixed_probabilities = kwargs['mixed_probabilities']

        if print_logs:
            print(f'creating mixed state after ramp')
            print(f'states: {mixed_basis_states}')
            print(f'probabilities: {mixed_probabilities}')

        mixed_state = 0
        for i in range(len(mixed_basis_states)):
            basis_state = qt.basis([current_measurement_simulation.num_levels]*current_measurement_simulation.num_qubits, [int(bit) for bit in mixed_basis_states[i]])
            mixed_state += mixed_probabilities[i] * basis_state * basis_state.dag()

        initial_state = mixed_state

    if kwargs.get('add_phase', False):
        pairs = kwargs['pairs']
        phases = kwargs['phases']

        if print_logs:
            print(f'adding phase ({list(np.round(np.array(phases)/np.pi, 3))} pi) to pairs ({pairs})')

        initial_state = add_phase(initial_state, current_measurement_simulation.number_operators, pairs, phases, print_logs=print_logs)

    if kwargs.get('add_x_rotation', False):
        pairs = kwargs['x_pairs']
        angles = kwargs['x_angles']

        if print_logs:
            print(f'adding x rotation ({list(np.round(np.array(angles)/np.pi, 3))} pi) to pairs ({pairs})')

        initial_state = add_x_rotation(initial_state, current_measurement_simulation.annihilation_operators, pairs, angles, print_logs=print_logs)


    if print_logs:
        if initial_state.isket:
            print(f'initial state after modifying is state vector')
        elif initial_state.isoper:
            print(f'initial state after modifying is density matrix')

    return initial_state



def add_phase(initial_state, number_operators, pairs, phases, print_logs=False):

    pair_1, pair_2 = pairs
    phase_1, phase_2 = phases

    number_operators = number_operators
    phase_matrix = 1j*phase_1/2*(number_operators[pair_1[1]] - number_operators[pair_1[0]])
    phase_matrix += 1j*phase_2/2*(number_operators[pair_2[1]] - number_operators[pair_2[0]])

    

    phase_unitary = phase_matrix.expm()

    if initial_state.isket:
        initial_state = phase_unitary * initial_state
    elif initial_state.isoper:
        initial_state = phase_unitary * initial_state * phase_unitary.dag()

    return initial_state

def add_x_rotation(initial_state, annihilation_operators, pairs, angles, print_logs=False):

    rotation_unitary = 1

    for i in range(len(pairs)):
        a_i = annihilation_operators[pairs[i][0]]
        a_j = annihilation_operators[pairs[i][1]]

        rotation_matrix = 1j*angles[i]/2*(a_i.dag()*a_j + a_j.dag()*a_i)
        rotation_unitary *= rotation_matrix.expm()

    if initial_state.isket:
        initial_state = rotation_unitary * initial_state
    elif initial_state.isoper:
        initial_state = rotation_unitary * initial_state * rotation_unitary.dag()

    return initial_state

def create_mixed_state(current_correlation_simulation, basis_states, probabilities, basis_state_to_index_dict, print_logs=False):
    # basis_state_to_index_dict = create_basis_state_to_index_mapping(pre_ramp_detuning, U)

    initial_state = 0
    for i in range(len(basis_states)):
        probability = probabilities[i]
        basis_state = list(int(bit) for bit in basis_states[i])

        '''
        convert a given pre-ramp basis state (str '0110' or list [0,1,1,0] e.g.) to the corresponding eigenstate of the 
        post-ramp Hamiltonian
        '''

        basis_state_index = basis_state_to_index_dict[tuple(basis_state)]
        particle_number_to_eigenstate_dict = current_correlation_simulation.get_particle_number_to_eigenstate_dict()
        num_particles = sum(basis_state)
        eigenstate = particle_number_to_eigenstate_dict[num_particles][basis_state_index]

        eigenstate_rho = eigenstate*eigenstate.dag()

        initial_state += probability * eigenstate_rho
    return initial_state


def create_basis_state_to_index_mapping(num_levels, num_qubits, pre_ramp_detuning, U):
    '''
    Creates a mapping from basis states to their index within the list of states with the same particle number
    '''

    all_basis_states = product(range(num_levels), repeat=num_qubits)

    particle_number_to_basis_states = {}
    particle_number_to_basis_state_energy = {}


    for basis_state in all_basis_states:
        energy = np.dot(basis_state, pre_ramp_detuning)

        for number in basis_state:
            energy += U*number*(number-1)

        particle_number = sum(basis_state)

        if not particle_number in particle_number_to_basis_states:
            particle_number_to_basis_states[particle_number] = []
            particle_number_to_basis_state_energy[particle_number] = []

        particle_number_to_basis_states[particle_number].append(basis_state)
        particle_number_to_basis_state_energy[particle_number].append(energy)

    basis_state_to_index = {}
    for particle_number in particle_number_to_basis_states:
        sorted_indices = np.argsort(particle_number_to_basis_state_energy[particle_number])
        particle_number_to_basis_state_energy[particle_number] = np.array(particle_number_to_basis_state_energy[particle_number])[sorted_indices]
        particle_number_to_basis_states[particle_number] = np.array(particle_number_to_basis_states[particle_number])[sorted_indices]

        for i in range(len(particle_number_to_basis_states[particle_number])):
            state = particle_number_to_basis_states[particle_number][i]
            basis_state_to_index[tuple(state)] = i

    return basis_state_to_index

def fit_to_data(simulation_kwargs, data, fit_params, simulation_data_generator):
    '''
    Fits simulation parameters to experimental or measured data by minimizing the difference 
    between simulated and measured values using a cost function.
    Parameters
    ----------
    simulation_kwargs : dict
        Dictionary containing initial values for simulation parameters. Expected keys include:
        'J', 'J_parallel', 'initial_detunings', 'scale_factor', 'phases'.
        Values for 'J', 'J_parallel', 'initial_detunings', and 'phases' should be array-like.
        'scale_factor' should be a float.
    data : array-like
        Experimental or measured data to which the simulation will be fitted.
    fit_params : list of str
        List of parameter names (keys from simulation_kwargs) to be optimized during fitting.
    simulation_data_generator : callable
        Function that generates simulated data given a set of parameters. Should accept the same
        keyword arguments as simulation_kwargs and return simulated data in the same format as `data`.
    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The result of the optimization process, containing optimized parameter values, 
        success flag, final cost, and other information as provided by `scipy.optimize.minimize`.
    Notes
    -----
    - The function constructs initial guesses and bounds for each parameter to be fitted.
    - The cost function used is the sum of squared errors between the measured and simulated data.
    - The optimization is performed using `scipy.optimize.minimize`.
    - Prints diagnostic information about the optimization process and the final fitted parameters.
    '''
    
    print(f'simulation_kwargs keys: {simulation_kwargs.keys()}')

    print(f'fit params: {fit_params}')

    initial_guess_dict = dict(
        J=simulation_kwargs['J'].copy(),
        J_parallel=simulation_kwargs['J_parallel'].copy(),
        initial_detunings=simulation_kwargs['initial_detunings'].copy(),
        scale_factor=simulation_kwargs['scale_factor'],
        phases=simulation_kwargs['phases'],
        x_angles=simulation_kwargs['x_angles'],
        measurement_J=simulation_kwargs['measurement_J'].copy(),
        measurement_J_parallel=simulation_kwargs['measurement_J_parallel'].copy()
    )

    

    J_delta_ratio = 0.8
    detuning_bound = 8 * 2 * np.pi  # 5 MHz

    bounds_dict = dict(
        J=[(val - np.abs(val) * J_delta_ratio, val + np.abs(val) * J_delta_ratio) for val in initial_guess_dict['J']],
        J_parallel=[(val - np.abs(val) * J_delta_ratio, val + np.abs(val) * J_delta_ratio) for val in initial_guess_dict['J_parallel']],
        initial_detunings=[(-detuning_bound, detuning_bound) for _ in initial_guess_dict['initial_detunings']],
        scale_factor=(0.2, 1),
        phases=[(-np.pi, np.pi) for _ in initial_guess_dict['phases']],
        x_angles=[(-np.pi, np.pi) for _ in initial_guess_dict['x_angles']]
    )

    bounds_dict['measurement_J'] = bounds_dict['J']
    bounds_dict['measurement_J_parallel'] = bounds_dict['J_parallel']



    def create_objective_function(_fit_params, _simulation_kwargs, _initial_guess_dict, _bounds_dict):

        key_to_indices = {}
        index_counter = 0

        initial_guess_list = []
        bounds_list = []

        for key in _simulation_kwargs:
            if key in _fit_params:
                initial_guess = _initial_guess_dict[key]
                bounds = _bounds_dict[key]

                if isinstance(bounds, list):
                    bounds_list.extend(bounds)
                else:
                    bounds_list.append(bounds)

                if isinstance(initial_guess, (list, tuple, np.ndarray)):
                    initial_guess_list.extend(initial_guess)

                    variable_length = len(initial_guess)
                    key_to_indices[key] = (index_counter, index_counter + variable_length)
                    index_counter += variable_length
                else:
                    initial_guess_list.append(initial_guess)

                    key_to_indices[key] = (index_counter, index_counter + 1)
                    index_counter += 1

        start_cutoff = simulation_kwargs.get('start_cutoff', 0)

        def objective_function(x):
            """
            Objective function to minimize.
            x is a flat array containing all parameters in the order defined by key_to_indices.
            """
            # Extract parameters based on key_to_indices
            params = simulation_kwargs.copy()
            for key, (start, end) in key_to_indices.items():
                params[key] = x[start:end] if end - start > 1 else x[start]

            # Calculate the N-terms data from the simulation
            populations_simulation = simulation_data_generator(**params)

            # Calculate the cost function as the sum squared error between measured and simulated N-terms
            # print(cost_function(data, populations_simulation))
            return cost_function(data[:,start_cutoff:], populations_simulation[:,start_cutoff:])

        return objective_function, key_to_indices, initial_guess_list, bounds_list


            

    objective_function, key_to_indices, initial_guess, bounds = create_objective_function(fit_params, simulation_kwargs, initial_guess_dict, bounds_dict)

    print(objective_function)
    print("Key to Indices:", key_to_indices)
    print("Initial Guess:", initial_guess)
    print("Bounds:", bounds)


    result = minimize(objective_function, initial_guess, bounds=bounds)
    print("Optimization success:", result.success)
    print("Final cost:", result.fun)

    for key in fit_params:
        start, end = key_to_indices[key]
        if end - start > 1:
            print(f"{key} = {list(result.x[start:end])}")
        else:
            print(f"{key} = {result.x[start]}")

    return result
