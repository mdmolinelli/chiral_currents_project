from itertools import product
import numpy as np
import qutip as qt

from src.src_current_measurement_simulations import CurrentMeasurementSimulation, generate_basis, create_annihilation_operators, generate_triangle_lattice_Hamiltonian, get_eigenstates_and_energies



class CurrentMeasurementSimulationReduced(CurrentMeasurementSimulation):
    '''
    Split the Hamiltonian into a reduced basis for the two pairs of readout qubits. We assume that
    these two pairs do not strongly interact with the other two pairs if there are more than 4 qubits
    in the simulation.
    '''
    def __init__(self, num_levels, num_qubits, num_particles, J, J_parallel, U, times, readout_pair_1, readout_pair_2, initial_detunings=None, 
                 measurement_detuning=None, measurement_J=None, measurement_J_parallel=None, psi0=None, time_offset=0, T1=None, T2=None, print_logs=False):
        self.reduced_indices = list(readout_pair_1) + list(readout_pair_2)
        self.reduced_indices_to_operator_index = {}
        for i in range(len(self.reduced_indices)):
            self.reduced_indices_to_operator_index[self.reduced_indices[i]] = i
    
        
        super().__init__(num_levels, num_qubits, num_particles, J, J_parallel, U, times, readout_pair_1, readout_pair_2, initial_detunings, 
                         measurement_detuning, measurement_J, measurement_J_parallel, psi0, time_offset, T1, T2, print_logs)

        if self.psi0.isket:
            psi0_density = self.psi0 * self.psi0.dag()
        elif self.psi0.isoper:
            psi0_density = self.psi0


        print(f'reduced indices: {self.reduced_indices}')
        print(psi0_density)
        psi0_density_reduced = psi0_density.ptrace(self.reduced_indices)
        self.psi0 = psi0_density_reduced

        
    def generate_triangle_lattice_Hamiltonian(self, annihilation_operators, J, J_parallel, U, detunings=None, initial=False):
        _detunings = detunings
        if initial:
            _annihilation_operators = self.annihilation_operators_full
            _J = J
            _J_parallel = J_parallel
        else:
            _annihilation_operators = annihilation_operators

            _J = np.zeros(len(self.reduced_indices)-1)
            _J_parallel = np.zeros(len(self.reduced_indices)-2)

            for i in range(len(self.reduced_indices)):
                for j in range(i+1, len(self.reduced_indices)):
                    J_ij = self.convert_index_pair_to_coupling(self.reduced_indices[i], self.reduced_indices[j])
                    if j == i + 1:
                        _J[i] = J_ij
                    elif j == i + 2:
                        _J_parallel[i] = J_ij


            if detunings is not None and isinstance(detunings, (list, np.ndarray)):
                _detunings = np.zeros(len(self.reduced_indices))
                for i in range(len(self.reduced_indices)):
                    _detunings[i] = detunings[self.reduced_indices[i]]

        return generate_triangle_lattice_Hamiltonian(_annihilation_operators, _J, _J_parallel, U, detunings=_detunings)

    
    def initialize_operators(self):


        self.annihilation_operators = create_annihilation_operators(self.num_levels, len(self.reduced_indices))
        self.annihilation_operators_full = create_annihilation_operators(self.num_levels, self.num_qubits)

        # number operators for each qubit
        self.number_operators = [op.dag() * op for op in self.annihilation_operators]
        self.number_operators_full = [op.dag() * op for op in self.annihilation_operators_full]

        # number correlators for the readout cross terms
        self.number_correlators = {}

        for i in range(len(self.readout_pair_1)):
            reduced_index_1 = self.readout_pair_1[i]
            # print(f'i: {i}, reduced index: {reduced_index_1}')
            for j in range(len(self.readout_pair_2)):
                reduced_index_2 = self.readout_pair_2[j]
                # print(f'j: {j}, reduced index: {reduced_index_2}')
                reduced_key = (i, j+2)
                # self.number_correlators[(reduced_index_1, reduced_index_2)] = self.number_operators[i] * self.number_operators[j+2]

                self.number_correlators[reduced_key] = self.number_operators[i] * self.number_operators[j+2]

        # print(f'number correlators: {self.number_correlators.keys()}')

        # current operators for the readout pairs
        self.current_operators = []

        for reduced_index_1, reduced_index_2 in self.readout_pairs:
            operator_index_1 = self.reduced_indices_to_operator_index[reduced_index_1]
            operator_index_2 = self.reduced_indices_to_operator_index[reduced_index_2]
            a_i = self.annihilation_operators[operator_index_1]
            a_j = self.annihilation_operators[operator_index_2]

            # print(f'reduced_index_1: {reduced_index_1}, reduced_index_2: {reduced_index_2}')
            # print(f'operator_index_1: {operator_index_1}, operator_index_2: {operator_index_2}')

            coupling = self.convert_index_pair_to_coupling(reduced_index_1, reduced_index_2)

            self.current_operators.append(1j*coupling*(a_i.dag() * a_j - a_j.dag() * a_i))

        self.current_correlator = self.current_operators[0] * self.current_operators[1]

    def get_particle_number_to_eigenenergy_dict(self):
        """
        Get the particle number to eigenenergy mapping.
        """
        if self.particle_number_to_eigenenergy is None:
            self.particle_number_to_eigenenergy, self.particle_number_to_eigenstate = get_eigenstates_and_energies(self.initial_Hamiltonian, self.number_operators_full)
        return self.particle_number_to_eigenenergy
    
    def get_particle_number_to_eigenstate_dict(self):
        """
        Get the particle number to eigenstate mapping.
        """
        if self.particle_number_to_eigenstate is None:
            self.particle_number_to_eigenenergy, self.particle_number_to_eigenstate = get_eigenstates_and_energies(self.initial_Hamiltonian, self.number_operators_full)
        return self.particle_number_to_eigenstate
    
    def get_population_average(self):
        """
        Get the average population for each qubit from the simulation result.
        """
        if self.population_average is None:
            result = self.get_simulation_result()
            self.population_average = np.array([qt.expect(op, result.states) for op in self.number_operators])
        return self.population_average
    
    def get_population_difference_average(self):
        """
        Get the average population difference for the readout pairs.
        """
        if self.population_difference_average is None:
            populations = self.get_population_average()
            self.population_difference_average = np.array([populations[index_2] - populations[index_1] for (index_1, index_2) in [[0,1],[2,3]]])
        return self.population_difference_average

    def get_covariance_sum(self):
        """
        Get the sum of the covariance for the readout pairs.
        """
        if self.covariance_sum is None:
            covariance = self.get_covariance()
            self.covariance_sum = np.zeros(covariance.shape[-1])

            self.covariance_sum += covariance[0, 2, :]
            self.covariance_sum -= covariance[0, 3, :]
            self.covariance_sum -= covariance[1, 2, :]
            self.covariance_sum += covariance[1, 3, :]

            self.covariance_sum *= np.sign(self.reduced_indices[1] - self.reduced_indices[0]) * np.sign(self.reduced_indices[3] - self.reduced_indices[2])

        
        return self.covariance_sum
    
    def get_current_correlation_from_operator(self):
        if self.covariance_sum_from_operator is None:
            self.covariance_sum_from_operator = qt.expect(self.current_correlator, self.get_states())

            self.covariance_sum_from_operator *= np.sign(self.reduced_indices[1] - self.reduced_indices[0]) * np.sign(self.reduced_indices[3] - self.reduced_indices[2])

        return self.covariance_sum_from_operator

def reduce_Hamiltonian(H, num_levels, num_qubits, qubit_indices):
    '''
    Reduce the Hamiltonian to only include the qubits in the readout pairs.
    '''

    basis = list(product(range(num_levels), repeat=num_qubits))
    new_basis = list(product(range(num_levels), repeat=len(qubit_indices)))

    H_reduced = np.zeros((len(new_basis), len(new_basis)), dtype=complex)

    for i in range(len(basis)):
        basis_state_1 = basis[i]
        # keep only matrix elements where the other qubits are in the ground state
        if not all(basis_state_1[k] == 0 for k in range(len(basis_state_1)) if k not in qubit_indices):
            continue
        new_index_1 = new_basis.index(tuple(np.array(basis_state_1)[qubit_indices]))
        for j in range(len(basis)):
            basis_state_2 = basis[j]
            # keep only matrix elements where the other qubits are in the ground state
            if not all(basis_state_2[k] == 0 for k in range(len(basis_state_2)) if k not in qubit_indices):
                continue
            new_index_2 = new_basis.index(tuple(np.array(basis_state_2)[qubit_indices]))
            H_reduced[new_index_1, new_index_2] = H[i, j]

    return qt.Qobj(H_reduced, dims=[[[num_levels]*len(qubit_indices)], [[num_levels]*len(qubit_indices)]])

