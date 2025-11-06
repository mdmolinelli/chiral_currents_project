import h5py
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize

class QubitOscillationMeasurement:

    def __init__(self, filename):
        self.filename = filename

        self.population = None
        self.population_corrected = None

        self.times = None

    def acquire_data(self):
        self.population, self.population_corrected, self.times, self.readout_list = acquire_data(self.filename)

    def get_population(self, corrected=False):
        if corrected:
            return self.get_population_corrected()
        if self.population is None:
            self.acquire_data()
        return self.population
    
    def get_population_corrected(self):
        if self.population_corrected is None:
            self.acquire_data()
        return self.population_corrected
    
    def get_times(self):
        if self.times is None:
            self.acquire_data()
        return self.times
    
    def plot_populations(self, corrected=False, both=False):
        times = self.get_times()
        population = self.get_population()

        if corrected or both:
            population_corrected = self.get_population_corrected()

        fig, axes = plt.subplots(population.shape[0], 1, figsize=(8, 2 * population.shape[0]), sharex=True)

        for i in range(population.shape[0]):
            if not corrected or both:
                axes[i].plot(times, population[i, :], 'o-', label='Population Average')
            if corrected or both:
                axes[i].plot(times, population_corrected[i, :], 'o-', label='Population Corrected')
            if both:
                axes[i].legend()


            axes[i].set_ylabel(f'Qubit {i+1}')
            axes[i].set_title(f'Population for Q{i+1}')
        
        axes[-1].set_xlabel('Time (ns)')
        plt.tight_layout()
        plt.show()

def acquire_data(filepath):

    time_units = 2.32515 / 16 # tproc_V1
    time_units = 2.32515*2 / 16 # tproc_V2

    with h5py.File(filepath, "r") as f:
        
        # for i in f:
            # print(f'{i}: {f[i][()]}')
            # print(i)

        population = f['population'][()]
        population_corrected = f['population_corrected'][()]
        times = f['expt_samples'][()]

        readout_list = [int(i) for i in f['readout_list'][()]]

    times *= time_units

    return population, population_corrected, times, readout_list

def generate_oscillation_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\QubitOscillations\QubitOscillations_{}\QubitOscillations_{}_{}_data.h5'.format(date_code, date_code, time_code)




def generate_triangle_lattice_Hamiltonian(annihilation_operators, J, J_parallel, U, detunings=None):
    """
    Generate the Hamiltonian for a triangle lattice with nearest neighbor interactions.

    Parameters:
    - annihilation_operators: List of annihilation operators for each qubit.
    - J: Coupling strength(s) between nearest neighbors. Can be a scalar or a list of length num_qubits-1.
    - J_parallel: Coupling strength(s) between parallel qubits. Can be a scalar or a list of length num_qubits-2.
    - U: On-site interaction strength.
    - detunings: Optional list of detuning values for each qubit.

    Returns:
    - Hamiltonian: Qutip Qobj representing the Hamiltonian of the system.
    """
    
    H = 0
    num_qubits = len(annihilation_operators)
    num_couplers = num_qubits - 2


    if isinstance(J, (int, float)):
        J = [J] * (num_qubits - 1)

    if not len(J) == num_qubits - 1:
        raise ValueError(f'Length of J array must be equal to the number of qubits minus 1 ({num_qubits - 1}), given: {len(J)}')

    if isinstance(J_parallel, (int, float)):
        J_parallel = [J_parallel] * (num_qubits - 2)

    if not len(J_parallel) == num_qubits - 2:
        raise ValueError(f'Length of J_parallel must be equal to the number of qubits minus 2 ({num_qubits - 2}), given: {len(J_parallel)}')
    
    # diagonal coupling
    for i in range(num_qubits):
        if i < num_qubits - 1:
            j = i + 1
            H += J[i] * (annihilation_operators[i].dag() * annihilation_operators[j] +
                        annihilation_operators[j].dag() * annihilation_operators[i])
    
    # parallel coupling
    for i in range(num_qubits):
        if i < num_qubits - 2:
            j = i + 2
            H += J_parallel[i] * (annihilation_operators[i].dag() * annihilation_operators[j] +
                                annihilation_operators[j].dag() * annihilation_operators[i])
    
    # On-site interactions
    for i in range(num_qubits):
        H += U * annihilation_operators[i].dag() * annihilation_operators[i] * (annihilation_operators[i].dag() * annihilation_operators[i] - 1)
    
    if detunings is not None:

        if isinstance(detunings, (int, float)):
            detunings = [detunings] * num_qubits

        if len(detunings) != num_qubits:
            raise ValueError("Detuning list must match the number of qubits.")
        for i, detuning in enumerate(detunings):
            H += detuning * annihilation_operators[i].dag() * annihilation_operators[i]
    
    return qt.Qobj(H)

def create_annihilation_operators(num_levels, num_qubits):
    """
    Create annihilation operators for a system of qubits.

    Parameters:
    - num_qubits: Number of qubits in the system.
    - num_levels: Number of energy levels for each qubit.

    Returns:
    - List of Qutip Qobj representing the annihilation operators for each qubit.
    """
    annihilation_operators = []
    for i in range(num_qubits):
        op_list = [qt.qeye(num_levels)]*num_qubits
        op_list[i] = qt.destroy(num_levels)
        a_i = qt.tensor(op_list)
        annihilation_operators.append(a_i)
    return annihilation_operators

def create_number_operators(annihilation_operators):
    """
    Create number operators for a system of qubits.

    Parameters:
    - annihilation_operators: List of annihilation operators for each qubit.

    Returns:
    - List of Qutip Qobj representing the number operators for each qubit.
    """
    number_operators = []
    for a in annihilation_operators:
        n = a.dag() * a
        number_operators.append(n)
    return number_operators

def create_collapse_operators(annihilation_operators, T1s=None, T2s=None):
    """
    Create collapse operators for a system of qubits.

    Parameters:
    - annihilation_operators: List of annihilation operators for each qubit.
    - gamma: Decay rate for the collapse operators.

    Returns:
    - List of Qutip Qobj representing the collapse operators for each qubit.
    """

    collapse_operators = []
    if not T1s is None:
        if isinstance(T1s, (int, float)):
            T1s = [T1s] * len(annihilation_operators)
        if len(T1s) != len(annihilation_operators):
            raise ValueError("Length of T1s must match the number of qubits.")

        gamma_1 = np.array([1 / T1 for T1 in T1s])

        if np.any(gamma_1 < 0):
            raise ValueError("T1 times must be positive.")


        for i in range(len(gamma_1)):
            c = np.sqrt(gamma_1[i]) * annihilation_operators[i]
            collapse_operators.append(c)
    else:
        gamma_1 = np.zeros(len(annihilation_operators))


    if not T2s is None:
        if isinstance(T2s, (int, float)):
            T2s = [T2s] * len(annihilation_operators)
        if len(T2s) != len(annihilation_operators):
            raise ValueError("Length of T2s must match the number of qubits.")

        gamma_2 = np.array([1 / T2 for T2 in T2s])
        gamma_phi = 2*gamma_2 - gamma_1

        if np.any(gamma_phi < 0):
            raise ValueError("T2 times must be no more than 2*T1.")

        for i in range(len(gamma_phi)):
            n = annihilation_operators[i].dag() * annihilation_operators[i]
            c = np.sqrt(gamma_phi[i]) * n
            collapse_operators.append(c)

    return collapse_operators

def calculate_population_simulation(annihilation_operators=None, number_operators=None, num_levels=None, num_qubits=None, J=None, J_parallel=None, U=None, T1s=None, T2s=None, times=None, detunings=None, psi0=None):
    collapse_operators = create_collapse_operators(annihilation_operators, T1s=T1s, T2s=T2s)
    H = generate_triangle_lattice_Hamiltonian(annihilation_operators, J, J_parallel, U, detunings=detunings)
    result = qt.mesolve(H, psi0, times, e_ops=number_operators, c_ops=collapse_operators)

    return np.array(result.expect)

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
        detunings=simulation_kwargs['detunings'].copy(),
        T1s=simulation_kwargs['T1s'].copy(),
        T2s=simulation_kwargs['T2s'].copy(),
    )

    

    J_delta_ratio = 0.8
    detuning_bound = 8 * 2 * np.pi  # 5 MHz

    bounds_dict = dict(
        J=[(val - np.abs(val) * J_delta_ratio, val + np.abs(val) * J_delta_ratio) for val in initial_guess_dict['J']],
        J_parallel=[(val - np.abs(val) * J_delta_ratio, val + np.abs(val) * J_delta_ratio) for val in initial_guess_dict['J_parallel']],
        detunings=[(-detuning_bound, detuning_bound) for _ in initial_guess_dict['detunings']],
        T1s=[(0.1, 100) for _ in initial_guess_dict['T1s']],
        T2s=[(0.1, 100) for _ in initial_guess_dict['T2s']],
    )


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

            # print(x)

            # Extract parameters based on key_to_indices
            params = simulation_kwargs.copy()
            for key, (start, end) in key_to_indices.items():
                params[key] = x[start:end] if end - start > 1 else x[start]

            # Calculate the N-terms data from the simulation
            populations_simulation = simulation_data_generator(**params)

            # Calculate the cost function as the sum squared error between measured and simulated N-terms
            # print(cost_function(data, populations_simulation))
            cost = np.sum(np.power((data[:,start_cutoff:], populations_simulation[:,start_cutoff:]), 2))
            # print(cost)
            return cost

        return objective_function, key_to_indices, initial_guess_list, bounds_list


            

    objective_function, key_to_indices, initial_guess, bounds = create_objective_function(fit_params, simulation_kwargs, initial_guess_dict, bounds_dict)

    print(objective_function)
    print("Key to Indices:", key_to_indices)
    print("Initial Guess:", initial_guess)
    print("Bounds:", bounds)


    result = minimize(objective_function, initial_guess, bounds=bounds, method='Nelder-Mead')
    print("Optimization success:", result.success)
    print("Final cost:", result.fun)

    for key in fit_params:
        start, end = key_to_indices[key]
        if end - start > 1:
            print(f"{key} = {list(result.x[start:end])}")
        else:
            print(f"{key} = {result.x[start]}")

    return result
