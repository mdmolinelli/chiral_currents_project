import numpy as np
import qutip as qt


from src.src_current_measurement_simulations import CurrentMeasurementSimulation, generate_basis

class CurrentMeasurementSimulationParticleSector(CurrentMeasurementSimulation):
    def __init__(self, num_levels, num_qubits, num_particles, J, J_parallel, U, times, readout_pair_1, readout_pair_2, initial_detunings=None, 
                 measurement_detuning=None, measurement_J=None, measurement_J_parallel=None, psi0=None, time_offset=0, T1=None, T2=None, print_logs=False):
        super().__init__(num_levels, num_qubits, num_particles, J, J_parallel, U, times, readout_pair_1, readout_pair_2, initial_detunings, 
                         measurement_detuning, measurement_J, measurement_J_parallel, psi0, time_offset, T1, T2, print_logs)

    def initialize_operators(self):
        '''
        Override to use particle sector operators
        '''

        self.single_particle_Hamiltonian = generate_triangle_ladder_single_particle_Hamiltonian(self.num_qubits, self.J_parallel, self.J, detuning=self.initial_detunings)

        self.basis = generate_basis(self.num_levels, self.num_qubits, self.num_particles)

        print(f'Number of basis states: {len(self.basis)}')
        print(self.basis)

        print('single particle Hamiltonian:')
        print(self.single_particle_Hamiltonian)

        # annihilation operators are not defined in the particle number sector
        self.annihilation_operators = []

        # number operators for each qubit
        self.number_operators = [np.zeros((len(self.basis), len(self.basis)), dtype=complex) for _ in range(self.num_qubits)]

        for i in range(len(self.basis)):
            state = self.basis[i]

            for j in range(len(state)):
                self.number_operators[j][i, i] = state[j]

        for i in range(len(self.number_operators)):
            self.number_operators[i] = qt.Qobj(self.number_operators[i])

        

        # number correlators for the readout cross terms
        self.number_correlators = {}

        for index_1 in self.readout_pair_1:
            for index_2 in self.readout_pair_2:
                self.number_correlators[(index_1, index_2)] = self.number_operators[index_1] * self.number_operators[index_2]

        # current operators for the readout pairs
        self.current_operators = [np.zeros((len(self.basis), len(self.basis)), dtype=complex) for _ in range(self.num_qubits-1)]

        for i in range(self.num_qubits-1):
            j = i + 1
            matrix_element = self.single_particle_Hamiltonian[i, j]
            print(f'matrix element {i},{j}: {matrix_element}')
            self.current_operators[i][i, j] = -1j * matrix_element
            self.current_operators[i][j, i] = 1j * np.conjugate(matrix_element)

            self.current_operators[i] = qt.Qobj(self.current_operators[i])


        min_readout_pair_1 = min(self.readout_pair_1)
        multiplier_1 = np.sign(self.readout_pair_1[1] - self.readout_pair_1[0])
        min_readout_pair_2 = min(self.readout_pair_2)
        multiplier_2 = np.sign(self.readout_pair_2[1] - self.readout_pair_2[0])
        self.current_correlator = multiplier_1 * multiplier_2 * self.current_operators[min_readout_pair_1] * self.current_operators[min_readout_pair_2]

   
    def generate_triangle_lattice_Hamiltonian(self, *args, **kwargs):
        '''
        Override to use particle sector operators
        '''
        return generate_triangle_lattice_Hamiltonian_particle_sector(self.num_qubits, self.num_particles, *args, **kwargs)

def generate_triangle_lattice_Hamiltonian_particle_sector(num_qubits, particle_number, annihilation_operators, J, J_parallel, U, detuning=None, **kwargs):
    """
    Construct the many-body Hamiltonian for bosons.
    """


    # print(f'detuning in generate_Hamiltonian: {detuning}')
    single_particle_hamiltonian = generate_triangle_ladder_single_particle_Hamiltonian(num_qubits=num_qubits, J_parallel=J_parallel, J_perp=J,
                                                                                       detuning=detuning)

    # print('single particle Hamiltonian')
    # print(single_particle_hamiltonian)

    if detuning is None:
        detuning = 0

    if isinstance(detuning, (int, float)):
        detuning = np.array([detuning] * num_qubits)
    

    # Generate basis and mapping to index
    basis = generate_basis(particle_number+1, num_qubits, particle_number)
    basis_to_index = {state: idx for idx, state in enumerate(basis)}
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)

    # Loop over basis states
    for state in basis:
        state_idx = basis_to_index[state]
        state_list = list(state)
        
        onsite_energy = 0

        # For each occupied site
        for i in range(num_qubits):

            onsite_energy += U/2*state_list[i]*(state_list[i] -1)  
        
            if state_list[i] >= 1:
                # onsite energy
                onsite_energy += state_list[i]*detuning[i]
            
                # For each possible hop to an empty site j
                for j in range(num_qubits):
                    if state_list[j] == 0 and single_particle_hamiltonian[i][j] != 0:
                        # Create new state by moving particle from i to j
                        new_state = state_list.copy()
                        new_state[i] -= 1
                        new_state[j] += 1
                        new_state = tuple(new_state)
                        if new_state in basis_to_index:
                            new_idx = basis_to_index[new_state]
                        
                            # Set matrix element
                            H[state_idx, new_idx] = single_particle_hamiltonian[i][j] * np.sqrt((new_state[i]+1)*new_state[j])
                            # For a Hermitian Hamiltonian, also set the symmetric element
                            H[new_idx, state_idx] = single_particle_hamiltonian[j][i] * np.sqrt((new_state[i]+1)*new_state[j])
    
        H[state_idx, state_idx] = onsite_energy
    
    # Diagonal onsite energies can be added here if needed.
    return qt.Qobj(H)




def generate_triangle_ladder_single_particle_Hamiltonian(num_qubits=None, J_parallel=None, J_perp=None, phase=None, detuning=None, periodic=False):
    """
    Generate the single-particle Hamiltonian for a triangle ladder system.
    """


    if detuning is None:
        detuning = 0

    if phase is None:
        phase = 0

    if isinstance(J_parallel, (int, float)):
        J_parallel = np.array([J_parallel] * (num_qubits - 2))
    if not len(J_parallel) == num_qubits - 2:
        raise ValueError("J_parallel must be a scalar or a list/array of length num_qubits - 2.")
    
    if isinstance(J_perp, (int, float)):
        J_perp = np.array([J_perp] * (num_qubits - 1))
    if not len(J_perp) == num_qubits - 1:
        raise ValueError("J_perp must be a scalar or a list/array of length num_qubits - 1.")

    

    if isinstance(phase, (int, float)):
        phase = np.array([phase] * (num_qubits - 2))

    if isinstance(detuning, (int, float)):
        detuning = np.array([detuning] * num_qubits)

    detuning -= np.min(detuning)  # shift detuning to have minimum at 0

    H = np.zeros((num_qubits, num_qubits), dtype='complex')

    for i in range(num_qubits):
        # qubit i is coupled to qubit i + 1 and qubit i + 2
        H[i, i] = detuning[i]

        if i < num_qubits - 1:
            H[i, i + 1] = J_perp[i]
            H[i + 1, i] = np.conjugate(J_perp[i])

        if i < num_qubits - 2:
            multiplier = -1
            if i % 2 == 1:
                multiplier = 1
            H[i, i + 2] = J_parallel[i] * np.exp(-1j * multiplier * phase[i])
            H[i + 2, i] = np.conjugate(J_parallel[i] * np.exp(-1j * multiplier * phase[i]))

    if periodic:
        if num_qubits >= 4 and num_qubits % 2 == 0:
            # qubit 0 is coupled to qubit num_qubits - 1 and qubit num_qubits - 2
            H[0, num_qubits - 1] = J_perp
            H[num_qubits - 1, 0] = np.conjugate(J_perp)
            
            H[0, num_qubits - 2] = J_parallel * np.exp(1j * phase[0])
            H[num_qubits - 2, 0] = np.conjugate(J_parallel * np.exp(1j * phase[0]))

            H[1, num_qubits - 1] = J_parallel * np.exp(1j * phase[1])
            H[num_qubits - 1, 1] = np.conjugate(J_parallel * np.exp(1j * phase[1]))
        else:
            raise ValueError("Periodic boundary conditions only work for even number of qubits >= 4.")

    # print('Hamiltonian:')
    # print(qt.Qobj(H))

    return qt.Qobj(H)
