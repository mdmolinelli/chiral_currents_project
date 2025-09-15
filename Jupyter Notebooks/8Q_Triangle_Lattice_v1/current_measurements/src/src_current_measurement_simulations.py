from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

class CurrentMeasurementSimulation:

    def __init__(self, num_levels, num_qubits, num_particles, J, J_parallel, U, times, readout_pair_1, readout_pair_2, initial_detunings=None, 
                 measurement_detuning=None, measurement_J=None, measurement_J_parallel=None, psi0=None, time_offset=0, T1=None, T2=None, print_logs=False):

        '''`
        :param times: Array of time points for the measurement.
        :param readout_pair_1: First pair of readout indices.
        :param readout_pair_2: Second pair of readout indices.
        :param singleshot_2Q_measurements: Dictionary containing single-shot two-qubit measurement results
            with tuples of readout indices as keys. Defaults to None.
        :param time_offset: Time offset to apply to the first set of readout indices in units of number of samples. Defaults to 0.
                            positive time offset means that the first readout pair beamsplitter time is earlier than the second readout pair beamsplitter time.
                '''
        
        self.print_logs = print_logs
        
        self.num_qubits = num_qubits
        self.num_levels = num_levels
        self.num_particles = num_particles


        if isinstance(J, (int, float)):
            J = [J] * (num_qubits - 1)
        self.J = J

        if measurement_J is None:
            measurement_J = J
        if isinstance(measurement_J, (int, float)):
            measurement_J = [measurement_J] * (num_qubits - 1)
        elif isinstance(measurement_J, np.ndarray) and len(measurement_J.shape) == 0:
            measurement_J = [float(measurement_J)] * (num_qubits - 1)
        self.measurement_J = measurement_J

        if isinstance(J_parallel, (int, float)):
            J_parallel = [J_parallel] * (num_qubits - 2)
        self.J_parallel = J_parallel

        if measurement_J_parallel is None:
            measurement_J_parallel = J_parallel
        if isinstance(measurement_J_parallel, (int, float)):
            measurement_J_parallel = [measurement_J_parallel] * (num_qubits - 2)
        elif isinstance(measurement_J_parallel, np.ndarray) and len(measurement_J_parallel.shape) == 0:
            measurement_J_parallel = [float(measurement_J_parallel)] * (num_qubits - 2)
        self.measurement_J_parallel = measurement_J_parallel
        
        self.U = U
        self.times = times

        if initial_detunings is None:
            initial_detunings = [0] * num_qubits
        self.initial_detunings = initial_detunings

        if measurement_detuning is None:
            measurement_detuning = 50 * max(self.J[0], self.J_parallel[0])
        if isinstance(measurement_detuning, (int, float)):
            # by default, set first set of readout qubits to the measurement detuning
            measurement_detuning_list = [0] * num_qubits
            for index in readout_pair_1:
                measurement_detuning_list[index] = measurement_detuning
            measurement_detuning = measurement_detuning_list

        self.measurement_detuning = measurement_detuning


        if T1 is None:
            self.gamma_1 = np.zeros(num_qubits)
        else:
            if isinstance(T1, (int, float)):
                T1 = np.array([T1] * num_qubits)
            self.gamma_1 = 2*np.pi/T1

        if T2 is None:
            self.gamma_2 = np.zeros(num_qubits)
        else:
            if isinstance(T2, (int, float)):
                T2 = np.array([T2] * num_qubits)
            self.gamma_2 = 2*np.pi/T2

        self.gamma_phi = self.gamma_2 - self.gamma_1/2


        self.readout_pair_1 = readout_pair_1
        self.readout_pair_2 = readout_pair_2
        self.readout_pairs = [readout_pair_1, readout_pair_2]

        self.time_offset = time_offset

        self.beamsplitter_times = {}

        self.particle_number_to_eigenenergy = None
        self.particle_number_to_eigenstate = None

        self.simulation_result = None
        self.states = None


        self.population_shots = None
        self.population_average = None

        self.population_difference_shots = None
        self.population_difference_average = None

        self.standard_deviation = None
        self.covariance = None
        self.covariance_sum = None
        self.covariance_sum_from_operator = None
        self.correlation = None

        self.current_from_operator = None

        # these define the four terms of the correlator <(n2-n1)(n4-n3)> = <n1n3> + <n2n4> - <n1n4> - <n2n3> 
        self.n1n3 = None
        self.n2n4 = None
        self.n1n4 = None
        self.n2n3 = None
        self.n_terms = {}

        self.n1n3_shots = None
        self.n2n4_shots = None
        self.n1n4_shots = None
        self.n2n3_shots = None

        self.n1n3_average = None
        self.n2n4_average = None
        self.n1n4_average = None
        self.n2n3_average = None


        self.current_correlation = None
        self.current_correlation_from_operator = None

        self.initialize_operators()

        self.annihilation_operators = create_annihilation_operators(num_levels, num_qubits)
        self.initial_Hamiltonian = generate_triangle_lattice_Hamiltonian(self.annihilation_operators, self.J, self.J_parallel, self.U, self.initial_detunings)

        self.measurement_Hamiltonian = generate_triangle_lattice_Hamiltonian(self.annihilation_operators, self.measurement_J, self.measurement_J_parallel, self.U, self.measurement_detuning)

        uncoupled_J = [0] * (self.num_qubits - 1)
        uncoupled_J_parallel = [0] * (self.num_qubits - 2)

        # the list J represents the following couplings (Q1-Q2, Q2-Q3, Q3-Q4, ...)
        # the list J_parallel represents the following couplings (Q1-Q3, Q2-Q4, ...)
        # Set only the couplings between the readout pairs to be non-zero
        for i, j in self.readout_pairs:
            # Diagonal coupling (nearest neighbor)
            if abs(i - j) == 1:
                uncoupled_J[min(i, j)] = self.measurement_J[min(i, j)]
            # Parallel coupling (next-nearest neighbor)
            elif abs(i - j) == 2:
                uncoupled_J_parallel[min(i, j)] = self.measurement_J_parallel[min(i, j)]

        self.uncoupled_Hamiltonian = generate_triangle_lattice_Hamiltonian(self.annihilation_operators, uncoupled_J, uncoupled_J_parallel, self.U, 0)


        if psi0 is None:
            eigenstates = self.get_eigenstates(self.num_particles)
            if len(eigenstates) > 0:
                psi0 = eigenstates[-1]

        if isinstance(psi0, qt.Qobj):
            self.psi0 = psi0
        elif isinstance(psi0, str):
            if psi0 == 'highest_single_particle':
                particle_number = 1
                eigenstates = self.get_eigenstates(particle_number)
                if len(eigenstates) > 0:
                    self.psi0 = eigenstates[-1]
        elif isinstance(psi0, int):
            eigenstates = self.get_eigenstates(self.num_particles)
            if -len(eigenstates) <= psi0 < len(eigenstates):
                self.psi0 = eigenstates[psi0]
        else:
            raise TypeError(f"psi0 must be a Qobj or a string, given {type(psi0)}.")

        # print(f'setting psi0 to: {self.psi0}')
        # print('initial state populations:')
        # for op in self.number_operators:
            # print(qt.expect(op, self.psi0))

        if self.print_logs:
            print(f'psi0 dims: {self.psi0.dims}')

            if self.psi0.isket:
                print('psi0 is a state vector')
            
            if self.psi0.isoper:
                print('psi0 is a density matrix')


    def initialize_operators(self):
        self.annihilation_operators = create_annihilation_operators(self.num_levels, self.num_qubits)

        # number operators for each qubit
        self.number_operators = [op.dag() * op for op in self.annihilation_operators]

        # number correlators for the readout cross terms
        self.number_correlators = {}

        for index_1 in self.readout_pair_1:
            for index_2 in self.readout_pair_2:
                self.number_correlators[(index_1, index_2)] = self.number_operators[index_1] * self.number_operators[index_2]

        # current operators for the readout pairs
        self.current_operators = []

        for index_1, index_2 in self.readout_pairs:
            a_i = self.annihilation_operators[index_1]
            a_j = self.annihilation_operators[index_2]

            coupling = self.convert_index_pair_to_coupling(index_1, index_2)


            self.current_operators.append(-1j*coupling*(a_i.dag() * a_j - a_j.dag() * a_i))

        self.current_correlator = self.current_operators[0] * self.current_operators[1]


    def convert_index_pair_to_coupling(self, index_1, index_2):
        """
        Convert a pair of indices to the corresponding coupling.
        """
        if abs(index_1 - index_2) == 1:
            return self.J[min(index_1, index_2)]
        elif abs(index_1 - index_2) == 2:
            return self.J_parallel[min(index_1, index_2)]
        else:
            raise ValueError(f"Invalid index pair: {index_1}, {index_2}. Indices must be separated by 1 or 2.")    

    def get_particle_number_to_eigenenergy_dict(self):
        """
        Get the particle number to eigenenergy mapping.
        """
        if self.particle_number_to_eigenenergy is None:
            self.particle_number_to_eigenenergy, self.particle_number_to_eigenstate = get_eigenstates_and_energies(self.initial_Hamiltonian, self.annihilation_operators)
        return self.particle_number_to_eigenenergy
    
    def get_particle_number_to_eigenstate_dict(self):
        """
        Get the particle number to eigenstate mapping.
        """
        if self.particle_number_to_eigenstate is None:
            self.particle_number_to_eigenenergy, self.particle_number_to_eigenstate = get_eigenstates_and_energies(self.initial_Hamiltonian, self.annihilation_operators)
        return self.particle_number_to_eigenstate

    def get_eigenstates(self, particle_number):
        """
        Get the eigenstate corresponding to a specific particle number. Sorted by energy
        """
        particle_number_to_eigenstate = self.get_particle_number_to_eigenstate_dict()
        particle_number_to_energy = self.get_particle_number_to_eigenenergy_dict()

        if particle_number in particle_number_to_eigenstate:
            eigenstates = particle_number_to_eigenstate[particle_number]
            eigenenergies = particle_number_to_energy[particle_number]
            sorted_indices = np.argsort(eigenenergies)
            return [eigenstates[i] for i in sorted_indices]
        else:
            raise ValueError(f"No eigenstate found for particle number {particle_number}.")
    
    def get_simulation_result(self):
        """
        Run the simulation and return the result.
        """
        if self.simulation_result is None:
            self.run_simulation()
        return self.simulation_result
    
    def get_states(self):
        """
        Get the states from the simulation result.
        """
        if self.states is None:
            self.states = self.get_simulation_result().states
        return self.states
    
    def get_population_average(self):
        """
        Get the average population for each qubit from the simulation result.
        """
        if self.population_average is None:
            result = self.get_simulation_result()
            self.population_average = np.array([qt.expect(op, result.states) for op in self.number_operators])
        return self.population_average
    
    def get_population_shots(self):
        """
        Get the population shots from the simulation result.
        """
        if self.population_shots is None:
            result = self.get_simulation_result()

        return self.population_shots

    def get_population_difference_average(self):
        """
        Get the average population difference for the readout pairs.
        """
        if self.population_difference_average is None:
            populations = self.get_population_average()
            self.population_difference_average = np.array([populations[pair[1]] - populations[pair[0]] for pair in self.readout_pairs])
        return self.population_difference_average
        
    def get_standard_deviation(self):
        """
        Get the standard deviation of the populations.
        """
        if self.standard_deviation is None:
            self.calculate_standard_deviation()
        return self.standard_deviation
    
    def get_covariance(self):
        """
        Get the covariance of the populations.
        """
        if self.covariance is None:
            self.calculate_covariance()
        return self.covariance

    def get_covariance_sum(self):
        """
        Get the sum of the covariance for the readout pairs.
        """
        if self.covariance_sum is None:
            covariance = self.get_covariance()
            self.covariance_sum = np.zeros(covariance.shape[-1])

            self.covariance_sum += covariance[self.readout_pair_1[0], self.readout_pair_2[0], :]
            self.covariance_sum -= covariance[self.readout_pair_1[1], self.readout_pair_2[0], :]
            self.covariance_sum -= covariance[self.readout_pair_1[0], self.readout_pair_2[1], :]
            self.covariance_sum += covariance[self.readout_pair_1[1], self.readout_pair_2[1], :]

        
        return self.covariance_sum

    def get_correlation(self):
        """
        Get the correlation for the readout pairs.
        """
        if self.correlation is None:
            covariance = self.get_covariance()
            self.correlation = np.zeros_like(covariance)
            for i in range(covariance.shape[0]):
                for j in range(covariance.shape[1]):
                    for t in range(covariance.shape[2]):
                        if covariance[i, i, t] * covariance[j, j, t] != 0:
                            self.correlation[i, j, t] = covariance[i, j, t] / np.sqrt(covariance[i, i, t] * covariance[j, j, t])
                        else:
                            self.correlation[i, j, t] = 0
        return self.correlation

    def get_current_from_operator(self):
        if self.current_from_operator is None:
            self.current_from_operator = np.array([qt.expect(op, self.get_states()) for op in self.current_operators])
        return self.current_from_operator

    def get_n1n3(self):
        if self.n1n3 is None:
            self.n1n3 = self.get_n_term(0, 2)
        return self.n1n3
    
    def get_n2n4(self):
        if self.n2n4 is None:
            self.n2n4 = self.get_n_term(1, 3)
        return self.n2n4
    
    def get_n1n4(self):
        if self.n1n4 is None:
            self.n1n4 = self.get_n_term(0, 3)
        return self.n1n4
    
    def get_n2n3(self):
        if self.n2n3 is None:
            self.n2n3 = self.get_n_term(1, 2)
        return self.n2n3
    
    def get_n_term(self, index_1, index_2):
        """
        Get the n-term for the given indices.
        """

        if not (index_1, index_2) in self.n_terms:

            if self.time_offset == 0:
                # no need for two time correlators
                self.n_terms[(index_1, index_2)] = qt.expect(self.number_correlators[(index_1, index_2)], self.get_states())
            else:
                print(f'calculating two time correlators')

                if (index_1, index_2) not in self.number_correlators:
                    raise ValueError(f"No n-term found for indices {index_1} and {index_2}.")
                
                # calculates the two time correlator A(t+tau)B(t) for negative offset
                # A(t)B(t+tau) (reverse = true) for positive offset
                reverse = self.time_offset > 0

                a_op = self.number_operators[index_1]
                b_op = self.number_operators[index_2]
                
                c_ops = []

                correlator = qt.correlation_2op_2t(self.simulation_Hamiltonian, self.psi0, self.times,
                                                    [0, abs(self.time_offset)], c_ops, a_op, b_op, reverse=reverse)
                
                
                self.n_terms[(index_1, index_2)] = correlator[:,-1]

        return self.n_terms[(index_1, index_2)]

    def get_current_correlation(self):
        """
        Get the current correlation for the readout pairs.
        """
        if self.current_correlation is None:
            current_correlation = 0
            for index_1 in self.readout_pair_1:
                for index_2 in self.readout_pair_2:
                    n_ij = self.get_n_term(index_1, index_2)
                    if (index_1 + index_2) % 2 == 0:
                        current_correlation += n_ij
                    else:
                        current_correlation -= n_ij
            self.current_correlation = current_correlation
            
        return self.current_correlation
    
    def get_covariance_sum_from_operator(self):
        if self.current_correlation_from_operator is None:
            self.current_correlation_from_operator = qt.expect(self.current_correlator, self.get_states())
            
            print(self.current_correlation_from_operator[0]/(self.J[0]*self.J[-1]))

            current_average_from_operator = [qt.expect(current_operator, self.get_states()) for current_operator in self.current_operators]
            self.current_correlation_from_operator -= current_average_from_operator[0]*current_average_from_operator[1]


        return self.current_correlation_from_operator

    def get_current_correlation_from_operator(self):
        if self.covariance_sum_from_operator is None:
            self.covariance_sum_from_operator = qt.expect(self.current_correlator, self.get_states())
        return self.covariance_sum_from_operator

    def get_beamsplitter_time(self):
        """
        Use the beamsplitter time of the readout pair that we did not shift
        """
        if self.time_offset > 0:
            return self.__get_beamsplitter_time(0)
        else:
            return self.__get_beamsplitter_time(1)
    
    def __get_beamsplitter_time(self, readout_pair_index):
        """
        Get the time for the beamsplitter operation for the given readout pair index.
        """

        if not readout_pair_index in self.beamsplitter_times:


            if readout_pair_index < 0 or readout_pair_index >= len(self.readout_pairs):
                raise ValueError(f"Invalid readout pair index: {readout_pair_index}. Must be between 0 or 1.")
            
            coupling = self.convert_index_pair_to_coupling(*self.readout_pairs[readout_pair_index])

            self.beamsplitter_times[readout_pair_index] = abs(np.pi/(4*coupling*1e6) * 1e6)  # Convert to microseconds

        return self.beamsplitter_times[readout_pair_index]


    def run_simulation(self, uncoupled=False):
        """
        Run the simulation.
        """
        if uncoupled:
            self.simulation_Hamiltonian = self.uncoupled_Hamiltonian
        else:
            self.simulation_Hamiltonian = self.measurement_Hamiltonian

        if self.print_logs:
            print(f'running simulation')

        if np.all(self.gamma_1 == 0) and np.all(self.gamma_phi == 0) and not self.psi0.isoper:
            if self.print_logs:
                print(f'running sesolve')
            self.simulation_result = qt.sesolve(self.simulation_Hamiltonian, self.psi0, self.times)
        else:
            if self.print_logs:
                print(f'running mesolve')
            c_ops = [np.sqrt(gamma) * op for gamma, op in zip(self.gamma_1, self.annihilation_operators)]
            c_ops += [np.sqrt(gamma) * op.dag()*op for gamma, op in zip(self.gamma_phi, self.annihilation_operators)]
            self.simulation_result = qt.mesolve(self.simulation_Hamiltonian, self.psi0, self.times, c_ops=c_ops)


    def calculate_standard_deviation(self):
        states = self.get_states()
        population = self.get_population_average()

        self.standard_deviation = np.zeros_like(population)
        for i in range(population.shape[0]):
            self.standard_deviation[i,:] = np.sqrt([qt.expect(self.number_operators[i]**2, states[t]) - np.power(population[i,t], 2) for t in range(len(states))])
    
    def calculate_covariance(self):
        states = self.get_states()
        population = self.get_population_average()

        self.covariance = np.zeros((self.num_qubits, self.num_qubits, len(self.times)))
        for i in range(self.covariance.shape[0]):
            for j in range(self.covariance.shape[1]):
                self.covariance[i,j,:] = [qt.expect(self.number_operators[i] * self.number_operators[j], states[t]) - population[i,t] * population[j,t] for t in range(len(states))]

    def print_initial_Hamiltonian(self):
        """
        Print the initial Hamiltonian.
        """
        print("Initial Hamiltonian:")
        print(convert_to_reduced_hamiltonian(self.initial_Hamiltonian, self.num_levels, self.num_qubits, 1))

    def print_measurement_Hamiltonian(self):
        """
        Print the measurement Hamiltonian.
        """
        print("Measurement Hamiltonian:")
        print(convert_to_reduced_hamiltonian(self.measurement_Hamiltonian, self.num_levels, self.num_qubits, 1))

    def print_uncoupled_Hamiltonian(self):
        """
        Print the uncoupled Hamiltonian.
        """
        print("Uncoupled Hamiltonian:")
        print(convert_to_reduced_hamiltonian(self.uncoupled_Hamiltonian, self.num_levels, self.num_qubits, 1))

    def plot_populations(self, average=True, shots=False, plot_beamsplitter_time=False):
        """
        Plot the population of each qubit.
        """
        if average:
            populations = self.get_population_average()
        elif shots:
            populations = self.get_population_shots()
        else:
            raise ValueError("Either average or shots must be True.")



        # times for first readout pair
        times_1 = np.copy(self.times)

        # times for second readout pair
        times_2 = np.copy(self.times)

        # print(f'time offset: {self.time_offset} samples')

        if self.time_offset > 0:
            times_1 += abs(self.time_offset)
        elif self.time_offset < 0:
            # print(f'Shift second readout pair times by: {abs(self.time_offset)} samples')
            times_2 += abs(self.time_offset)


        times = [times_1, times_2]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        readout_indices = self.readout_pair_1 + self.readout_pair_2
        for idx, qubit_idx in enumerate(readout_indices):
            ax = axs[idx // 2, idx % 2]
            ax.plot(times[idx // 2], populations[qubit_idx], label=f'Q{qubit_idx+1}')
            ax.set_title(f'Qubit {qubit_idx+1} Population')
            ax.set_xlabel('Time')
            ax.set_ylabel('Population')
            ax.grid()
            ax.legend()

            if plot_beamsplitter_time:
                beamsplitter_time = self.get_beamsplitter_time()
                # print(f'beamsplitter_time: {beamsplitter_time}')
                ax.axvline(x=beamsplitter_time + abs(self.time_offset), color='r', linestyle='--', label='Beamsplitter Time')

        plt.tight_layout()
        plt.show()

    def plot_population_difference(self, average=True, shots=False, plot_beamsplitter_time=False):
        population_difference = self.get_population_difference_average()

        fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharey=True)
        for i, diff in enumerate(population_difference):
            axs[i].plot(self.times, diff)
            axs[i].set_xlabel('Time (μs)')
            axs[i].set_ylabel('Population Difference')
            axs[i].set_title(f'Population Difference {i+1}')
            axs[i].grid()

            if plot_beamsplitter_time:
                beamsplitter_time = self.get_beamsplitter_time()
                axs[i].axvline(x=beamsplitter_time + abs(self.time_offset), color='r', linestyle='--', label='Beamsplitter Time')
                axs[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_currents(self):
        currents = self.get_currents()

        for i in range(len(currents)):
            plt.plot(self.times, currents[i], label=f'Q{self.readout_pairs[i][0]+1}-Q{self.readout_pairs[i][1]+1}')

        plt.xlabel('Time (μs)')
        plt.ylabel('Current')
        plt.title('Currents')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_n_terms(self, expectation=True, shots_average=False, plot_beamsplitter_time=False):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        tlist = self.times

        n_terms = [self.get_n1n3(), self.get_n2n4(), self.get_n1n4(), self.get_n2n3()]
        labels = 'n1n3', 'n2n4', 'n1n4', 'n2n3'
        axes = axs.flatten()

        

        for i in range(len(n_terms)):

            ax = axes[i]
            ax.plot(tlist, n_terms[i], label=labels[i])
            ax.set_title(labels[i])
            ax.set_xlabel('Time')
            ax.set_ylabel('Expectation Value')
            ax.grid()
            ax.legend()

            if plot_beamsplitter_time:
                beamsplitter_time = self.get_beamsplitter_time()
                ax.axvline(x=beamsplitter_time, color='r', linestyle='--', label='Beamsplitter Time')

        plt.tight_layout()
        plt.show()


    def plot_current_correlation(self, expectation=True, shots_average=False, plot_beamsplitter_time=False):
        tlist = self.times

        current_correlation = self.get_current_correlation()

        plt.plot(tlist, current_correlation)
        plt.xlabel('Time')
        plt.ylabel('Current Correlation')
        plt.grid()
        plt.legend()
        plt.title('Current Correlation')

        # assume the same beamsplitter time for this plot
        beamsplitter_time = self.get_beamsplitter_time()
        plt.axvline(x=beamsplitter_time, color='r', linestyle='--', label='Beamsplitter Time')

        plt.tight_layout()
        plt.show()

    def plot_eigenstates(self, particle_number=None):
        """
        Plot the eigenstates for a given particle number.
        """
        
        if isinstance(particle_number, int):
            particle_number = [particle_number]

        if particle_number is None:
            particle_number = range(self.num_qubits + 1)

        particle_number_to_eigenstate = self.get_particle_number_to_eigenstate_dict()
        particle_number_to_eigenenergy = self.get_particle_number_to_eigenenergy_dict()


        for num in particle_number:
            if num in particle_number_to_eigenstate:
                for i in range(len(particle_number_to_eigenstate[num])):
                    eigenstate = particle_number_to_eigenstate[num][i]
                    eigenstate_energy = particle_number_to_eigenenergy[num][i]

                    qubit_occupation = np.array([qt.expect(op, eigenstate) for op in self.number_operators])

                    plt.figure(figsize=(10, 6))
                    plt.bar(range(self.num_qubits), qubit_occupation, tick_label=[f'Q{i+1}' for i in range(self.num_qubits)])
                    plt.title(f'Qubit Occupation for eigenstate #{i+1} with {num} particles\nEnergy: {eigenstate_energy:.2f}')
                    plt.xlabel('Qubit')
                    plt.ylabel('Occupation Probability')
                    plt.grid()
                    plt.show()

            else:
                print(f"No eigenstates found for particle number {num}.")

    def plot_standard_deviation(self):
        """
        Plot the standard deviation of the populations.
        """
        standard_deviation = self.get_standard_deviation()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for i in range(self.num_qubits):
            ax = axs[i // 2, i % 2]
            ax.plot(self.times, standard_deviation[i], label=f'Q{i+1}')
            ax.set_title(f'Standard Deviation Qubit {i+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Standard Deviation')
            ax.grid()
            ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_covariance(self):
        """
        Plot the covariance between each pair of qubits (Qi, Qj) where i < j.
        """
        covariance = self.get_covariance()

        # Calculate total number of subplots (upper triangle, excluding diagonal)
        num_plots = self.num_qubits * (self.num_qubits - 1) // 2
        num_cols = 2
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows), sharex=True, sharey=True)
        axs = axs.flatten()

        plot_idx = 0
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                ax = axs[plot_idx]
                ax.plot(self.times, covariance[i, j], label=f'Cov(Q{i+1}, Q{j+1})')
                ax.set_title(f'Q{i+1}, Q{j+1}')
                ax.legend()
                plot_idx += 1

        # Hide unused axes
        for k in range(plot_idx, len(axs)):
            axs[k].axis('off')

        
        plt.tight_layout()
        plt.show()

    def plot_correlation(self):
        """
        Plot the correlation between each pair of qubits (Qi, Qj) where i < j.
        """
        correlation = self.get_correlation()

        # Calculate total number of subplots (upper triangle, excluding diagonal)
        num_plots = self.num_qubits * (self.num_qubits - 1) // 2
        num_cols = 2
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows), sharex=True, sharey=True)
        axs = axs.flatten()

        plot_idx = 0
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                ax = axs[plot_idx]
                ax.plot(self.times, correlation[i, j], label=f'Corr(Q{i+1}, Q{j+1})')
                ax.set_title(f'Q{i+1}, Q{j+1}')
                ax.legend()
                plot_idx += 1

        # Hide unused axes
        for k in range(plot_idx, len(axs)):
            axs[k].axis('off')

        
        plt.tight_layout()
        plt.show()


    def plot_covariance_sum(self):
        """
        Plot the sum of the covariance for the readout pairs.
        """
        covariance_sum = self.get_covariance_sum()

        plt.figure(figsize=(10, 6))
        plt.plot(self.times, covariance_sum, label='Covariance Sum')
        plt.xlabel('Time')
        plt.ylabel('Covariance Sum')
        plt.title('Sum of Covariance for Readout Pairs')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()



def get_eigenstates_and_energies(H, annihilation_operators):
    ### find eigenstates and eigenenergies of resonant Hamiltonian

    # call this on the qutip Qobj hamiltonian H_qobj (replace with your variable)
    import numpy as np, scipy.linalg as sla, faulthandler, os
    faulthandler.enable(all_threads=True)
 
    ##############
    def validate_and_test_eigh(H_qobj, save_prefix="bad_matrix"):
        # Get dense matrix as numpy array
        H = H_qobj.full() if hasattr(H_qobj, "full") else H_qobj.data.toarray()
        H = np.asarray(H, dtype=np.complex128)
        print("shape:", H.shape, "dtype:", H.dtype, "contiguous:", H.flags['C_CONTIGUOUS'])

        # Check finite
        nfinite = np.isfinite(H).sum()
        print("finite entries:", nfinite, "/", H.size)
        if not np.all(np.isfinite(H)):
            print("Non-finite entries detected")
            np.save(save_prefix + "_nonfinite.npy", H)
            return False

        # Check Hermitian
        herm_error = np.max(np.abs(H - H.conj().T))
        print("max|H - H^†| =", herm_error)
        if herm_error > 1e-10:
            print("Warning: H not Hermitian (max diff > 1e-10)")
            np.save(save_prefix + "_not_herm.npy", H)
            # continue to attempt eigh but note the mismatch

        for k in range(1000):
            try:
                w, v = np.linalg.eigh(H)   # or sla.eigh(H) if you prefer to test SciPy path
            except Exception as e:
                print("failed at iteration", k, "with", e)
                break
        else:
            print("no failure in 1000 iterations")


        # Try direct scipy.linalg.eigh in try/except and log
        try:
            w, v = sla.eigh(H, overwrite_a=False, check_finite=True)
            print("eigh succeeded; first eigenvalue:", w[0])
            return True
        except Exception as e:
            print("scipy.linalg.eigh raised exception:", repr(e))
            # Still possible to segfault; call with faulthandler active will provide trace
            return False

    #############

    # validate_and_test_eigh(H)

    
    particle_number_to_eigenenergy = {}
    particle_number_to_eigenstate = {}

    eigenenergies, eigenstates = H.eigenstates()

    total_number_operator = np.sum([op.dag() * op for op in annihilation_operators])

    for i in range(len(eigenenergies)):
        eigenenergy = eigenenergies[i]
        eigenstate = eigenstates[i]

        particle_number = np.round(qt.expect(total_number_operator, eigenstate))

        if particle_number not in particle_number_to_eigenenergy:
            particle_number_to_eigenenergy[particle_number] = []
            particle_number_to_eigenstate[particle_number] = []

        particle_number_to_eigenenergy[particle_number].append(eigenenergy)
        particle_number_to_eigenstate[particle_number].append(eigenstate)

    return particle_number_to_eigenenergy, particle_number_to_eigenstate

def create_annihilation_operators(num_levels, num_qubits):
    """
    Create a list of annihilation operators for a system with a specified number of levels and qubits.
    
    Parameters:
    num_levels (int): Number of energy levels for each qubit.
    num_qubits (int): Number of qubits in the system.
    
    Returns:
    list: A list of annihilation operators for each qubit.
    """

    annihilation_operators = []

    a = qt.destroy(num_levels)

    for i in range(num_qubits):
        annihilation_operators.append(qt.tensor([a if j == i else qt.qeye(num_levels) for j in range(num_qubits)]))

    return annihilation_operators


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


def generate_basis(num_levels, num_qubits, num_particles):
    basis = [state[::-1] for state in product(range(num_levels), repeat=num_qubits) if sum(state) == num_particles]
    return basis



def convert_to_reduced_hamiltonian(hamiltonian, num_levels, num_qubits, num_particles):
    """
    Convert a Hamiltonian to a reduced form for a system with a specified number of levels and qubits.
    
    Parameters:
    hamiltonian (qutip.Qobj): The original Hamiltonian.
    num_levels (int): Number of energy levels for each qubit.
    num_qubits (int): Number of qubits in the system.
    num_particles (int): Number of particles in the system.
    
    Returns:
    qutip.Qobj: The reduced Hamiltonian.
    """

    basis = generate_basis(num_levels, num_qubits, num_particles)

    # print('basis:')
    # for state in basis:
        # print(state)

    basis_vectors = [qt.basis([num_levels] * num_qubits, list(state)) for state in basis]

    # Project Hamiltonian onto the reduced basis
    projectors = [vec for vec in basis_vectors]
    reduced_dim = len(projectors)
    reduced_hamiltonian = np.zeros((reduced_dim, reduced_dim), dtype=complex)

    for i, ket_i in enumerate(projectors):
        for j, ket_j in enumerate(projectors):
            reduced_hamiltonian[i, j] = (ket_i.dag() * hamiltonian * ket_j)


    reduced_hamiltonian_qobj = qt.Qobj(reduced_hamiltonian, dims=[[reduced_dim], [reduced_dim]])
    
    return reduced_hamiltonian_qobj
    
