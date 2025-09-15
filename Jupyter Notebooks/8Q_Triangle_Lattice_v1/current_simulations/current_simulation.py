import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, product
import qutip as qt
from scipy.optimize import curve_fit

from quantum_states import QuantumState, FockBasisState

class CurrentSimulation:
    def __init__(self, num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U, detuning=None, periodic=False):
        """
        Initialize the CurrentSimulations class with default parameters.
        
        :param num_levels: Number of levels for each qubit.
        :param num_qubits: Number of qubits for the many-body Hamiltonian.
        :param num_particles: Number of particles for the many-body Hamiltonian.
        :param J_parallel: Coupling strength for parallel connections.
        :param J_perp: Coupling strength for perpendicular connections.
        :param phase: phase of flux through each plaqette, either provided as a list of floats or a single float, in units of pi
        :param detuning: Detuning values for each qubit (can be a scalar or an array).
        :param U: Onsite interaction strength.
        :param periodic: Whether the system has periodic boundary conditions or not.
        """

        self.num_levels = num_levels
        self.num_qubits = num_qubits
        self.num_particles = num_particles
        self.num_states = math.comb(num_qubits, num_particles)

        self.J_parallel = J_parallel
        self.J_perp = J_perp

        if isinstance(phase, (int, float)):
            phase = [phase] * num_qubits
        self.phase = phase


        self.detuning = detuning
        self.U = U
        self.periodic = periodic


        self.basis = None

        # memoize single particle Hamiltonian for provided detunings
        self.single_particle_hamiltonian_dict = {}
        self.resonant_Hamiltonian = None
        self.off_resonant_Hamiltonian = None

        self.eigenenergies = None
        self.eigenstates = None

        self.currents = None
        self.current_correlations = None

        self.total_chiral_current = None
        self.average_rung_current = None

        self.__init_current_operator()

    def get_basis(self):
        return generate_basis(self.num_qubits, self.num_particles, self.num_levels)
        
    def get_single_particle_Hamiltonian_matrix_element(self, i, j):
        return self.get_single_particle_Hamiltonian()[i, j]

    def get_single_particle_Hamiltonian(self, detuning=None):
        """
        Get the single-particle Hamiltonian for a triangle ladder system.
        """
        return generate_triangle_ladder_single_particle_Hamiltonian(self.num_qubits, self.J_parallel, self.J_perp, self.phase, detuning, self.periodic)

    def get_resonant_Hamiltonian(self):
        """
        Get the many-body Hamiltonian for resonant hardcore bosons.
        """
        return self.generate_resonant_Hamiltonian()
    
    def get_off_resonant_Hamiltonian(self):
        """
        Get the many-body Hamiltonian for off-resonant hardcore bosons.
        """
        return self.generate_off_resonant_Hamiltonian()
    
    def generate_resonant_Hamiltonian(self):
        """
        Generate the many-body Hamiltonian for resonant hardcore bosons
        """
        return self.__generate_Hamiltonian(detuning=0)
    
    def generate_off_resonant_Hamiltonian(self):
        """
        Generate the many-body Hamiltonian for off-resonant hardcore bosons
        """
        return self.__generate_Hamiltonian(detuning=self.detuning)

    def __generate_Hamiltonian(self, detuning=None):
        """
        Construct the many-body Hamiltonian for hardcore bosons.
        """

        # print(f'detuning in generate_Hamiltonian: {detuning}')
        single_particle_hamiltonian = self.get_single_particle_Hamiltonian(detuning=detuning)

        # print('single particle Hamiltonian')
        # print(single_particle_hamiltonian)

        if detuning is None:
            detuning = 0

        if isinstance(detuning, (int, float)):
            detuning = np.array([detuning] * self.num_qubits)
     

        # Generate basis and mapping to index
        basis = self.get_basis()
        basis_to_index = {state: idx for idx, state in enumerate(basis)}
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=complex)

        # Loop over basis states
        for state in basis:
            state_idx = basis_to_index[state]
            state_list = list(state)
            
            onsite_energy = 0

            # For each occupied site
            for i in range(self.num_qubits):

                onsite_energy += self.U/2*state_list[i]*(state_list[i] -1)  
            
                if state_list[i] >= 1:
                    # onsite energy
                    onsite_energy += state_list[i]*detuning[i]
                
                    # For each possible hop to an empty site j
                    for j in range(self.num_qubits):
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
    
    def get_resonant_ground_state(self):
        """
        Get the ground state for the resonant hardcore bosons.
        """
        return self.get_resonant_excited_state(0)
    
    def get_resonant_excited_state(self, excited_state_index):
        """
        Get the excited state for the resonant hardcore bosons.
        """
        return self.get_resonant_eigenstates()[excited_state_index]
    
    def get_resonant_eigenstates(self):
        eigenenergies, eigenstates = self.get_resonant_Hamiltonian().eigenstates()
        return eigenstates
    
    def get_resonant_eigenenergies(self):
        eigenenergies, eigenstates = self.get_resonant_Hamiltonian().eigenstates()
        return eigenenergies
    

    def get_ground_state(self):
        """
        Get the ground state for the hardcore bosons.
        """
        return self.get_excited_state(0)
    
    def get_excited_state(self, excited_state_index):
        """
        Get the excited state for the hardcore bosons.
        """
        return self.get_eigenstates()[excited_state_index]
    
    def get_eigenstates(self):
        eigenenergies, eigenstates = self.get_off_resonant_Hamiltonian().eigenstates()
        return eigenstates
    
    def get_eigenenergies(self):
        eigenenergies, eigenstates = self.get_off_resonant_Hamiltonian().eigenstates()
        return eigenenergies

    
    def run_simulation(self, psi0, tlist, resonant=False):
        """
        Run the simulation for the current parameters.
        """

        # Run the simulation
        self.populations = np.zeros((self.num_qubits, len(tlist)))


        try:
            if resonant:
                resonant_Hamiltonian = self.get_resonant_Hamiltonian()
                result = qt.sesolve(resonant_Hamiltonian, psi0, tlist)
                # print(self.get_resonant_Hamiltonian())

            else:
                off_resonant_Hamiltonian = self.get_off_resonant_Hamiltonian()
                result = qt.sesolve(off_resonant_Hamiltonian, psi0, tlist)
                # print(self.get_off_resonant_Hamiltonian())
        except Exception as e:
            print("Error during simulation:", e)
            print(f"start time: {tlist[0]}")
            print(f"stop time: {tlist[-1]}")
            print(f"number of points: {len(tlist)}")

            return None
        else:

            self.psi0 = psi0
            self.tlist = tlist
            self.result = result
            

            basis = self.get_basis()
            for i in range(len(tlist)):
                state_vector = result.states[i].data.to_array()[:,0]
                for j in range(len(basis)):
                    amp = state_vector[j]
                    for k in range(len(basis[j])):
                        if basis[j][k] == 1:
                            self.populations[k,i] += np.power(np.abs(amp), 2)

            # for i in range(populations.shape[0]):

            #     plt.plot(tlist, populations[i,:], label=f'Q{i+1}')
                
            # plt.ylim(-0.1, 1.1)
                
            # plt.xlabel('time ($\mu$s)')
            # plt.ylabel('Populations')
            # plt.legend()
            # plt.show()

            return result
    
    def get_populations(self):
        """
        Get the populations of the qubits.
        """
        if not hasattr(self, 'result'):
            raise ValueError("Please run a simulation first.")
        return self.populations

    def plot_populations(self):
        """
        Plot the populations of the qubits.
        """
        if not hasattr(self, 'result'):
            raise ValueError("Please run a simulation first.")
        
        if self.result is None:
            print('No simulation result available.')
            return

        # Plot the populations
        e_ops = []
        for i in range(self.num_states):
            number_operator = np.zeros((num_states,num_states))
            number_operator[i,i] = 1
            e_ops.append(qt.Qobj(number_operator))

        for i in range(self.num_qubits):
            plt.plot(self.tlist, [qt.expect(e_ops[i], self.result.states[j]) for j in range(len(self.result.states))], label=f"Qubit {i}")

        plt.legend()
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel("Populations")
        plt.show()

    def __init_current_operator(self):
        a = qt.destroy(self.num_levels)
        a_2Q_1 = qt.tensor(a, qt.qeye(self.num_levels))
        a_2Q_2 = qt.tensor(qt.qeye(self.num_levels), a)

        ### 2Q current operator
        self.current_operator_2Q = lambda J: 1j*(J*a_2Q_1.dag()*a_2Q_2 - np.conjugate(J)*a_2Q_2.dag()*a_2Q_1)

        ### 3Q composite state operators
        a_3Q_1 = qt.tensor(a, qt.qeye(self.num_levels), qt.qeye(self.num_levels))
        a_3Q_2 = qt.tensor(qt.qeye(self.num_levels), a, qt.qeye(self.num_levels))
        a_3Q_3 = qt.tensor(qt.qeye(self.num_levels), qt.qeye(self.num_levels), a)

        self.current_operator_3Q_12 = lambda J: 1j*(J*a_3Q_1.dag()*a_3Q_2 - np.conjugate(J)*a_3Q_2.dag()*a_3Q_1)
        self.current_operator_3Q_23 = lambda J: 1j*(J*a_3Q_2.dag()*a_3Q_3 - np.conjugate(J)*a_3Q_3.dag()*a_3Q_2)
        self.current_operator_3Q_13 = lambda J: 1j*(J*a_3Q_1.dag()*a_3Q_3 - np.conjugate(J)*a_3Q_3.dag()*a_3Q_1)

        ### 4Q composite state operators
        self.current_operator_4Q_1 = lambda J: qt.tensor(self.current_operator_2Q(J), qt.tensor(qt.qeye(self.num_levels), qt.qeye(self.num_levels)))
        self.current_operator_4Q_2 = lambda J: qt.tensor(qt.tensor(qt.qeye(self.num_levels), qt.qeye(self.num_levels)), self.current_operator_2Q(J))

    def calculate_current(self, density_matrix, J):
        return (density_matrix * self.current_operator_2Q(J)).tr()

    def calculate_current_correlation(self, density_matrix, J_12, J_34):
        return (density_matrix * self.current_operator_4Q_1(J_12) * self.current_operator_4Q_2(J_34)).tr()

    def calculate_3Q_current_correlation(self, density_matrix, J_12, J_13, J_23):
        '''
        in the case of the 3Q system, with edges that share a vertex, the current operator and composite state are different
        '''
        
        current_correlation_12_23 = (density_matrix * self.current_operator_3Q_12(J_12) * self.current_operator_3Q_23(J_23)).tr()
        current_correlation_12_13 = (density_matrix * self.current_operator_3Q_12(J_12) * self.current_operator_3Q_13(J_13)).tr()
        current_correlation_23_13 = (density_matrix * self.current_operator_3Q_23(J_23) * self.current_operator_3Q_13(J_13)).tr()
        
        return current_correlation_12_23, current_correlation_12_13, current_correlation_23_13

    def calculate_currents(self, psi0):
        all_qubits = list(range(1, self.num_qubits+1))
        all_qubit_pairs = list(combinations(all_qubits, 2))

        full_psi0 = convert_reduced_to_full_state(self.num_qubits, self.num_particles, self.num_levels, psi0)
        currents = {}

        # calculate currents
        for i in range(len(all_qubit_pairs)):
            
            q_1, q_2 = all_qubit_pairs[i]

            coupling = self.get_single_particle_Hamiltonian()[q_1 - 1, q_2 - 1]
            if coupling == 0:
                continue

            print(q_1, q_2)
            
            rho = full_psi0.ptrace([q_1-1, q_2-1])
            
            currents[q_1, q_2] = self.calculate_current(rho, coupling).real/2/np.pi
            
        self.currents = currents

    def calculate_current_correlations(self, psi0):
        # calculate correlations

        print('calculating current correlations with initial state:')
        print(psi0)

        all_qubits = list(range(1, self.num_qubits+1))
        all_qubit_pairs = list(combinations(all_qubits, 2))
        
        full_psi0 = convert_reduced_to_full_state(self.num_qubits, self.num_particles, self.num_levels, psi0)
        current_correlations = {}

        for i in range(len(all_qubit_pairs)):
            q_11, q_12 = all_qubit_pairs[i]
            coupling_1 = self.get_single_particle_Hamiltonian()[q_11 - 1, q_12 - 1]
            print(q_11, q_12)
            
            for j in range(i+1, len(all_qubit_pairs)):
                q_21, q_22 = all_qubit_pairs[j]        
                coupling_2 = self.get_single_particle_Hamiltonian()[q_21 - 1, q_22 - 1]

                if coupling_1 == 0 or coupling_2 == 0:
                    continue

                
                if q_21 in [q_11, q_12] or q_22 in [q_11, q_12]:
                    # need to do 3Q system
                    rho = full_psi0.ptrace(list(set([q_11-1, q_12-1, q_21-1, q_22-1])))
                    # q_11, q_12 should always be leg 12
                        # unless q_12 and q_22 are the same
                    # if q_11 and q_21 are the same, then q_21 - q_22 should be leg 13
                    # if q_12 and q_21 are the same, then q_21 - q_22 should be leg 23
                    # if q_12 and q_22 are the same, then q_11 - q_12 should be leg 13 and q_21 - q_22 should be leg 23
                    if q_11 == q_21:
                        current_correlation_12_23, current_correlation_12_13, current_correlation_23_13 = self.calculate_3Q_current_correlation(rho, coupling_1, coupling_2, 0)
                        current_correlations[((q_11, q_12),(q_21, q_22))] = current_correlation_12_13.real/2/np.pi/2/np.pi
                    elif q_12 == q_21:
                        current_correlation_12_23, current_correlation_12_13, current_correlation_23_13 = self.calculate_3Q_current_correlation(rho, coupling_1, 0, coupling_2)
                        current_correlations[((q_11, q_12),(q_21, q_22))] = current_correlation_12_23.real/2/np.pi/2/np.pi
                    elif q_12 == q_22:
                        current_correlation_12_23, current_correlation_12_13, current_correlation_23_13 = self.calculate_3Q_current_correlation(rho, 0, coupling_1, coupling_2)
                        current_correlations[((q_11, q_12),(q_21, q_22))] = current_correlation_23_13.real/2/np.pi/2/np.pi
                    else:
                        # print('other case')
                        # print((q_11, q_12),(q_21, q_22))
                        pass
                else:
                    rho = full_psi0.ptrace([q_11-1, q_12-1, q_21-1, q_22-1])
                    current_correlations[((q_11, q_12),(q_21, q_22))] = self.calculate_current_correlation(rho, coupling_1, coupling_2).real/2/np.pi/2/np.pi
                
        self.current_correlations = current_correlations

    def calculate_currents_fock(self, psi0):
        # current operator is iJ_ij(a_i^dagger a_j - a_j^dagger a_i)

        all_qubits = list(range(1, self.num_qubits+1))
        all_qubit_pairs = list(combinations(all_qubits, 2))

        psi0_fock = convert_reduced_to_fock_state(self.get_basis(), psi0)
        currents = {}

        # calculate currents
        # define the leg current operator as j_j,m = −iJ(exp(iχ(−1)^m b†_j,m b_j+1,m − H.c.)
        # let's define the phase through each plaquette as chi. Let's put all the phase in the J_parallel coupling, so rung coupling is the same but with chi = 0
        # for example, for a right side up plaquette, j_j,1 = −iJ(exp(-iχ b†_j,1 b_j+1,1 − H.c.)
        # for example, for a upside down plaquette, j_j,2 = −iJ(exp(iχ b†_j,2 b_j+1,2 − H.c.)
        for i in range(len(all_qubit_pairs)):
            
            q_1, q_2 = all_qubit_pairs[i]


            # phase information stored in coupling in Hamiltonian matrix element
            coupling = self.get_single_particle_Hamiltonian()[q_1 - 1, q_2 - 1]
            if coupling == 0:
                continue

            value = 0
            # a_i^dagger a_j term
            new_state = psi0_fock.apply_raising_operator(q_1-1, self.num_levels).apply_lowering_operator(q_2-1)
            value += psi0_fock.inner_product(new_state) * -1j * coupling
            # a_j^dagger a_i term
            new_state = psi0_fock.apply_lowering_operator(q_1-1).apply_raising_operator(q_2-1, self.num_levels)
            value -= psi0_fock.inner_product(new_state) * -1j * np.conjugate(coupling)

            # for rung currents starting at top rung, (odd qubits) multiply by -1
            if abs(q_1 - q_2) == 1 and q_1 % 2 == 1:
                value *= -1

            currents[q_1, q_2] = value.real/2/np.pi
            # break

        self.currents = currents


    def calculate_current_correlations_fock(self, psi0):
        # calculate correlations
        # current operator is -J_ij(a_i^dagger a_j - a_j^dagger a_i)*J_kl(a_k^dagger a_l - a_l^dagger a_k)


        # print('calculating current correlations with initial state:')
        # print(psi0)

        all_qubits = list(range(1, self.num_qubits+1))
        all_qubit_pairs = list(combinations(all_qubits, 2))
        
        basis = self.get_basis()
        psi0_fock = convert_reduced_to_fock_state(basis, psi0)
        current_correlations = {}

        # define the leg current operator as j_j,m = −iJ(exp(iχ(−1)^m b†_j,m b_j+1,m − H.c.)
        # let's define the phase through each plaquette as chi. Let's put all the phase in the J_parallel coupling, so rung coupling is the same but with chi = 0
        # for example, for a right side up plaquette, j_j,1 = −iJ(exp(-iχ b†_j,1 b_j+1,1 − H.c.)
        # for example, for a upside down plaquette, j_j,2 = −iJ(exp(iχ b†_j,2 b_j+1,2 − H.c.)
        for i in range(len(all_qubit_pairs)):
            q_11, q_12 = all_qubit_pairs[i]
            # phase information stored in coupling in Hamiltonian matrix element
            coupling_1 = self.get_single_particle_Hamiltonian()[q_11 - 1, q_12 - 1]
            # print(q_11, q_12)
            
            for j in range(i+1, len(all_qubit_pairs)):
                q_21, q_22 = all_qubit_pairs[j]        
                # phase information stored in coupling in Hamiltonian matrix element
                coupling_2 = self.get_single_particle_Hamiltonian()[q_21 - 1, q_22 - 1]

                if coupling_1 == 0 or coupling_2 == 0:
                    continue


                value = 0
                # a_i^dagger a_j a_k^dagger a_l
                new_state = psi0_fock.apply_raising_operator(q_11-1, self.num_levels).apply_lowering_operator(q_12-1).apply_raising_operator(q_21-1, self.num_levels).apply_lowering_operator(q_22-1)
                value -= psi0_fock.inner_product(new_state) * coupling_1 * coupling_2

                # a_i^dagger a_j a_l^dagger a_k
                new_state = psi0_fock.apply_raising_operator(q_11-1, self.num_levels).apply_lowering_operator(q_12-1).apply_raising_operator(q_22-1, self.num_levels).apply_lowering_operator(q_21-1)
                value += psi0_fock.inner_product(new_state) * coupling_1 * np.conjugate(coupling_2)

                # a_j^dagger a_i a_k^dagger a_l
                new_state = psi0_fock.apply_raising_operator(q_12-1, self.num_levels).apply_lowering_operator(q_11-1).apply_raising_operator(q_21-1, self.num_levels).apply_lowering_operator(q_22-1)
                value += psi0_fock.inner_product(new_state) * np.conjugate(coupling_1) * coupling_2

                # a_j^dagger a_i a_l^dagger a_k
                new_state = psi0_fock.apply_raising_operator(q_12-1, self.num_levels).apply_lowering_operator(q_11-1).apply_raising_operator(q_22-1, self.num_levels).apply_lowering_operator(q_21-1)
                value -= psi0_fock.inner_product(new_state) * np.conjugate(coupling_1) * np.conjugate(coupling_2)

                # for rung currents starting at top rung, (odd qubits) multiply by -1
                if abs(q_11 - q_12) == 1 and q_11 % 2 == 1:
                    value *= -1
                if abs(q_21 - q_22) == 1 and q_21 % 2 == 1:
                    value *= -1

                current_correlations[((q_11, q_12),(q_21, q_22))] = value.real/2/np.pi/2/np.pi

        self.current_correlations = current_correlations

    def get_currents(self):
        self.calculate_currents_fock(self.psi0)
        return self.currents

    def get_current_correlations(self):
        self.calculate_current_correlations_fock(self.psi0)
        return self.current_correlations
    
    def calculate_total_chiral_current(self, psi0=None):
        """
        Calculate the total chiral current for a given initial state.
        """
        # Calculate the total chiral current

        if psi0 is None:
            psi0 = self.get_resonant_ground_state()

        psi0_fock = convert_reduced_to_fock_state(self.get_basis(), psi0)

        # print('calculating total chiral current with initial state:')
        # print(psi0_fock)

        chiral_current = 0

        # J_c = 1/(N - 1) * sum_j <j_j,1 - j_j,2>
        # j_j,m = -i J_parallel (exp(i chi (-1)^m) a^dag_j,m a_j+1,m - h.c)
        for i in range(self.num_qubits-2):


            q_1 = i + 1
            q_2 = i + 3

            phase = 0
            # if abs(q_1 - q_2) == 1:
            #     phase = 0
            # elif abs(q_1 - q_2) == 2:
            #     # j = 1,3,5,... is upper leg
            #     # j = 2,4,6,... is lower leg
            #     phase = -1j * self.phase[q_1 - 1] * (-1)**(q_1 % 2)


            coupling = self.get_single_particle_Hamiltonian()[q_1 - 1, q_2 - 1]
            if coupling == 0:
                continue

            value = 0
            # a_i^dagger a_j term
            new_state = psi0_fock.apply_raising_operator(q_1-1, self.num_levels).apply_lowering_operator(q_2-1)
            value += psi0_fock.inner_product(new_state) * 1j * coupling
            # a_j^dagger a_i term
            new_state = psi0_fock.apply_lowering_operator(q_1-1).apply_raising_operator(q_2-1, self.num_levels)
            value -= psi0_fock.inner_product(new_state) * 1j * np.conjugate(coupling)
            

            # for bottom rung, (even qubits) multiply by -1
            if q_1 % 2 == 0:
                value *= -1

            print(q_1, q_2, value.real/2/np.pi)
            chiral_current += value.real/2/np.pi

        self.total_chiral_current = chiral_current / (self.num_qubits - 2)

    def calculate_average_rung_current(self, psi0):
        """
        Calculate the average rung current for a given initial state.
        """
        # Calculate the total chiral current

        print('calculating average rung current')

        if psi0 is None:
            psi0 = self.get_resonant_ground_state()

        psi0_fock = convert_reduced_to_fock_state(self.get_basis(), psi0)

        rung_current = 0

        # J_c = 1/(N - 1) * sum_j |<j_j>|
        # j_j = -i J (a^dag_j,1 a_j+1,2 - h.c)
        for i in range(self.num_qubits-1):


            q_1 = i + 1
            q_2 = i + 2

            coupling = self.get_single_particle_Hamiltonian()[q_1 - 1, q_2 - 1]
            if coupling == 0:
                continue

            value = 0
            # a_i^dagger a_j term
            new_state = psi0_fock.apply_raising_operator(q_1-1, self.num_levels).apply_lowering_operator(q_2-1)
            value += psi0_fock.inner_product(new_state) * 1j * coupling
            # a_j^dagger a_i term
            new_state = psi0_fock.apply_lowering_operator(q_1-1).apply_raising_operator(q_2-1, self.num_levels)
            value -= psi0_fock.inner_product(new_state) * 1j * coupling


            # for diagonal down to the right, (odd qubits) multiply by -1
            if q_1 % 2 == 1:
                value *= -1
            
            print(q_1, q_2, value.real/2/np.pi)
            rung_current += np.abs(value.real/2/np.pi)

        self.average_rung_current = rung_current / (self.num_qubits - 1)

    def calculate_density_imbalance(self, psi0=None):
        print('calculating density imbalance')

        if psi0 is None:
            psi0 = self.get_resonant_ground_state()

        psi0_fock = convert_reduced_to_fock_state(self.get_basis(), psi0)

        density_imbalance = 0

        # Delta n = 1/N * sum_j <n_j,1 - n_j,2>
        # j_j = -i J (a^dag_j,1 a_j+1,2 - h.c)
        for i in range(self.num_qubits):

            q = i + 1

            # a_i^dagger a_i term
            new_state = psi0_fock.apply_lowering_operator(q-1).apply_raising_operator(q-1, self.num_levels)
            value =  psi0_fock.inner_product(new_state)

            # even qubit, upper leg
            if q % 2 == 0:
                value *= -1

            density_imbalance += value.real

        self.density_imbalance = density_imbalance / self.num_qubits
        
    def get_total_chiral_current(self, psi0=None):
        if psi0 is None:
            psi0 = self.psi0
        self.calculate_total_chiral_current(psi0)
        return self.total_chiral_current
    

    def get_average_rung_current(self, psi0=None):
        if psi0 is None:
            psi0 = self.psi0
        self.calculate_average_rung_current(psi0)
        return self.average_rung_current
    
    def get_density_imbalance(self, psi0=None):
        if psi0 is None:
            psi0 = self.psi0
        self.calculate_density_imbalance(psi0)
        return self.density_imbalance

    def calculate_center_rung_correlations(self, psi0):
        '''
        calculate the correlations from the rung at the center to all other rungs
        '''


    def get_center_rung_correlations(self, psi0=None):
        if psi0 is None:
            psi0 = self.psi0

        # TODO: I don't like how this is setting psi0 here, either make psi0 a parameter to get current correlations
        # or completely redo how the class stores calculations
        self.psi0 = psi0
        current_correlations = self.get_current_correlations()


        center_rung_correlations = np.zeros(self.num_qubits - 2, dtype='complex')
        # center_rung_correlations = {}
        for key, value in current_correlations.items():
            q_11, q_12 = key[0]
            q_21, q_22 = key[1]
            # check if the first pair is the center rung
            if q_11 == self.num_qubits//2 and q_12 == self.num_qubits//2 + 1:
                # check if the second pair is also a rung
                if abs(q_22 - q_21) == 1:
                    # subtract by 2 to account for skipping the center rung 
                    # (index is one less for rungs after the center rung)
                    center_rung_correlations[q_21-2] = value

            # center rung can also be second pair
            if q_21 == self.num_qubits//2 and q_22 == self.num_qubits//2 + 1:
                # check if the second pair is also a rung
                if abs(q_12 - q_11) == 1:
                    center_rung_correlations[q_11-1] = value

        print(f'center rung correlations: {center_rung_correlations}')
        self.center_rung_correlations = center_rung_correlations
        return self.center_rung_correlations


def generate_triangle_ladder_single_particle_Hamiltonian(num_qubits=None, J_parallel=None, J_perp=None, phase=None, detuning=None, periodic=False):
    """
    Generate the single-particle Hamiltonian for a triangle ladder system.
    """

    if detuning is None:
        detuning = 0

    if phase is None:
        phase = 0

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
            H[i, i + 1] = J_perp
            H[i + 1, i] = np.conjugate(J_perp)

        if i < num_qubits - 2:
            multiplier = -1
            if i % 2 == 1:
                multiplier = 1
            H[i, i + 2] = J_parallel * np.exp(-1j * multiplier * phase[i])
            H[i + 2, i] = np.conjugate(J_parallel * np.exp(-1j * multiplier * phase[i]))

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

def generate_basis(N, M, num_levels):
    """
    Generate all bitstrings of length N with M ones.
    """

    basis = [state for state in product(range(num_levels), repeat=N) if sum(state) == M]
    return basis

def convert_fock_to_reduced_state(reduced_basis, fock_state):

    reduced_state = np.zeros(len(reduced_basis), dtype='complex')

    for i in range(len(reduced_basis)):
        reduced_basis_state = reduced_basis[i]
        if reduced_basis_state in fock_state.tuple_to_coeff:
            reduced_state[i] = fock_state.tuple_to_coeff[reduced_basis_state]

    return qt.Qobj(reduced_state, dims=(len(reduced_basis), 1))


def convert_reduced_to_fock_state(reduced_basis, reduced_state):
    fock_state = []
    fock_basis = []
    for i in range(len(reduced_basis)):
        fock_basis.append(FockBasisState(reduced_basis[i]))
        fock_state.append(reduced_state[i,0])

    return QuantumState(fock_state, fock_basis)

def convert_reduced_to_full_state(num_qubits, num_particles, num_levels, reduced_state):
    '''
    convert an N qubit state in the M particle to an N qubit state in 2^N space
    '''
    
    reduced_basis = generate_basis(num_qubits, num_particles, num_levels)
    full_basis = [bits for bits in product(range(num_levels), repeat=num_qubits)]
    
    full_state = np.zeros(num_levels**num_qubits, dtype='complex')
    
    for i in range(len(reduced_basis)):
        reduced_basis_state = reduced_basis[i]
        
        full_basis_index = full_basis.index(reduced_basis_state)
        
        full_state[full_basis_index] = reduced_state[i]
    
    return qt.Qobj(full_state, dims=[[num_levels] * num_qubits, [1]*num_qubits])


if __name__ == "__main__":
    num_levels = 2
    num_qubits = 4
    num_particles = 2

    num_states = math.comb(num_qubits, num_particles)

    J_parallel = 1 * 2 * np.pi
    J_perp = J_parallel
    phase = 0
    U = 1 * 2 * np.pi

    detuning = [1000]*num_qubits
    detuning[0] = 0
    detuning[1] = 0


    simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U)

    psi0 = simulation.get_resonant_ground_state()
    print(f'psi0: {psi0}')
    times = np.linspace(0, 1, 101)

    result = simulation.run_simulation(psi0, times, resonant=False)
    # simulation.plot_populations()


    # test current correlations
    # correlations with 4 - 2 system
    # correlations with 4 - 2 system
    positive_current_state = 1/2*(qt.Qobj([0, 1, 0, 0, 0, 0]) + 1j*qt.Qobj([0, 0, 1, 0, 0, 0]) + 1j*qt.Qobj([0, 0, 0, 1, 0, 0]) - qt.Qobj([0, 0, 0, 0, 1, 0]))
    negative_current_state = 1/2*(qt.Qobj([0, 1, 0, 0, 0, 0]) - 1j*qt.Qobj([0, 0, 1, 0, 0, 0]) - 1j*qt.Qobj([0, 0, 0, 1, 0, 0]) - qt.Qobj([0, 0, 0, 0, 1, 0]))
    counter_current_state = 1/np.sqrt(2)*(positive_current_state + negative_current_state)


    # List of states to evaluate
    states = {
        "Positive Current State": positive_current_state,
        "Negative Current State": negative_current_state,
        "Counter Current State": counter_current_state
    }

    # Calculate and print currents and correlations
    for state_name, state in states.items():
        print(f"\n{state_name}:")
        
        # Calculate currents
        simulation.psi0 = state
        currents = simulation.get_currents()
        print("Currents:")
        for qubit_pair, current in currents.items():
            print(f"  Qubits {qubit_pair}: {current:.6f}")
        
        # Calculate correlations
        simulation.psi0 = state
        correlations = simulation.get_current_correlations()
        print("Correlations:")
        for qubit_pairs, correlation in correlations.items():
            print(f"  Qubit pairs {qubit_pairs}: {correlation:.6f}")


    # compare to julia simulations
    num_levels = 4
    num_qubits = 3
    num_particles = 1

    num_states = math.comb(num_qubits, num_particles)

    J_parallel = 0.5
    J_perp = 1
    phase = 0.4 * np.pi
    U = 2.5

    simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U)
    psi0 = simulation.get_resonant_ground_state()
    simulation.psi0 = psi0

    print('test of 3 qubits, 1 particle, 4 levels to compare to Julia simulations\n')

    data = psi0.data.to_array()
    print("state populations should be [0.26839597570409707, 0.4632080485918, 0.26839597570410323]")
    print(np.power(np.abs(data[:,0]),2))
    print()

    currents = simulation.get_currents()
    print("currents should be [0.2470363884970871, 0.2470363884970871, -0.2470363884970871]")
    print({key: 2*np.pi*currents[key] for key in currents})


    # compare to current correlations
    print('test of 4 qubits, 1 particle, 3 levels to compare to Julia simulations\n')


    num_levels = 3
    num_qubits = 4
    num_particles = 1

    num_states = math.comb(num_qubits, num_particles)

    J_parallel = 0.5
    J_perp = 1
    phase = 0.4 * np.pi
    U = 2.5

    simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U)
    psi0 = simulation.get_resonant_ground_state()
    simulation.psi0 = psi0

    print('test of 3 qubits, 1 particle, 4 levels to compare to Julia simulations\n')

    data = psi0.data.to_array()
    print("state populations should be [0.22673278 0.02816136 0.54847568 0.19663018]")
    print(np.power(np.abs(data[:,0]),2))
    print()

    # current_correlations = simulation.get_current_correlations()
    # print("currents correlations (relative to (1,2)) should be [0.2470363884970871, 0.2470363884970871, -0.2470363884970871]")
    # print({key: 2*np.pi*currents[key] for key in currents})


    # print('\n\ntesting\n')
    # # print('hamiltonian:')
    # # print(simulation.get_single_particle_Hamiltonian())
    # print(psi0)
    # psi0_fock = convert_reduced_to_fock_state(simulation.get_basis(), psi0)
    # print(f'psi_fock: {psi0_fock}')



    ### check current correlations for our system parameters

    # check correlations relative to (1,2) for both 0 and pi flux and fit correlation lengths
    # sweep J_parallel/J_perp

    num_levels = 4
    num_qubits = 8
    num_particles = 4

    num_states = math.comb(num_qubits, num_particles)

    J_parallel = -1 * 2 * np.pi
    J_perp = -1 * 2 * np.pi
    U = 20 * 2 * np.pi


    J_parallels = np.linspace(0, 5, 11)
    run_sweep = False


    # first check for one J_parallel value
    correlations_0 = np.zeros((num_qubits - 3))
    correlations_pi = np.zeros((num_qubits - 3))

    print('current correlations for J_parallel/J_perp = 1 should be:')
    print('[0.320285381498738, 0.15301274847738278, 0.08693222283493184, 0.06484964239491028, 0.05968977033329019]')


    # 0 flux
    simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, 0, U)
    psi0 = simulation.get_resonant_ground_state()
    simulation.psi0 = psi0

    current_correlations = simulation.get_current_correlations()


    # [ 0.32410091 -0.19289472  0.12671357 -0.09933339  0.08468678]

    for pair_1, pair_2 in current_correlations:
        if pair_1 == (1, 2):
            if abs(pair_2[1] - pair_2[0]) == 1 and pair_2[0] > 2:
                index = pair_2[0] - 3
                # print(pair_2, current_correlations[(pair_1, pair_2)])
                correlations_0[index] = current_correlations[(pair_1, pair_2)]

    # pi flux
    simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, np.pi, U)
    psi0 = simulation.get_resonant_ground_state()
    simulation.psi0 = psi0

    current_correlations = simulation.get_current_correlations()

    for pair_1, pair_2 in current_correlations:
        if pair_1 == (1, 2):
            if abs(pair_2[1] - pair_2[0]) == 1 and pair_2[0] > 2:
                index = pair_2[0] - 3
                # print(pair_2, current_correlations[(pair_1, pair_2)])
                correlations_pi[index] = current_correlations[(pair_1, pair_2)]

    print('correlations_0: ', correlations_0)
    print('correlations_pi: ', correlations_pi)

    plt.plot(correlations_0, linestyle=':', marker='o', color='blue', ms=8, label='0 flux')
    plt.plot(correlations_pi, linestyle='', marker='o', color='red', ms=8, label='pi flux')
    plt.xlabel('rung index')
    plt.ylabel('current correlation')
    plt.title('Current Correlations')
    plt.legend()
    plt.show()

    # now sweep J_parallel
    if run_sweep:

        coherence_lengths = np.zeros(len(J_parallels))
        correlations_0 = np.zeros((len(J_parallels), num_qubits - 3))
        correlations_pi = np.zeros((len(J_parallels), num_qubits - 3))

        for i in range(len(J_parallels)):
            J_parallel = J_parallels[i]*J_perp

            # 0 flux
            simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, 0, U)
            psi0 = simulation.get_resonant_ground_state()
            simulation.psi0 = psi0

            current_correlations = simulation.get_current_correlations()


            for pair_1, pair_2 in current_correlations:
                if pair_1 == (1, 2):
                    if abs(pair_2[1] - pair_2[0]) == 1 and pair_2[0] > 2:
                        index = pair_2[0] - 3
                        # print(pair_2, current_correlations[(pair_1, pair_2)])
                        correlations_0[i,index] = current_correlations[(pair_1, pair_2)]

            # pi flux
            simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, np.pi, U)
            psi0 = simulation.get_resonant_ground_state()
            simulation.psi0 = psi0

            current_correlations = simulation.get_current_correlations()

            for pair_1, pair_2 in current_correlations:
                if pair_1 == (1, 2):
                    if abs(pair_2[1] - pair_2[0]) == 1 and pair_2[0] > 2:
                        index = pair_2[0] - 3
                        # print(pair_2, current_correlations[(pair_1, pair_2)])
                        correlations_pi[i,index] = current_correlations[(pair_1, pair_2)]

            
            # plt.plot(correlations_0, linestyle=':', marker='o', color='blue', ms=8, label='0 flux')
            # plt.plot(correlations_pi, linestyle='', marker='o', color='red', ms=8, label='pi flux')
            # plt.plot(np.abs(correlations_pi), linestyle=':', marker='o', color='red', ms=8, label='pi flux magnitude')

            # plt.xlabel('rung index')
            # plt.ylabel('current correlation')
            # plt.title('Current Correlations')
            # plt.legend()
            # plt.show()

            print(f'correlations_0: {correlations_0[i,:]}')
            print(f'correlations_pi: {correlations_pi[i,:]}')


            def exp_fit(x, a, b, c):
                return a * np.exp(-x/b) + c
            
            initial_guess = (correlations_pi[i,0], 2.2, 0)
            bounds = ([0, 0, -np.inf], [10, np.inf, np.inf])

            try:
                popt, pcov = curve_fit(exp_fit, np.array(range(len(correlations_pi[i,:]))), 
                                    np.abs(correlations_pi[i,:]), p0=(1, 0.1, 0), bounds=bounds)
            except RuntimeError as e:
                print(f"Error in curve fitting: {e}")
                popt = [correlations_pi[i,0], 0, 0]

                fit_points = np.linspace(0, len(correlations_pi[i,:])*1.2, 101)
                plt.plot(np.abs(correlations_pi[i,:]), marker='o', linestyle='', color='blue', ms=8, label=f'data')
                plt.plot(fit_points, exp_fit(fit_points, *initial_guess), color='red', label='guess')
                plt.xlabel('rung index')
                plt.ylabel('current correlation')
                plt.title(f'Current correlations for J_parallel/J_perp = {J_parallel/J_perp:.2f}')
                plt.show()

            else:
                print(f'curve fit success')
                print(f'fit params: {popt}')

                

                # Map index i to a color from the color scheme
                color = mpl.colormaps['viridis'](i / len(J_parallels))

                fit_points = np.linspace(0, len(correlations_pi[i,:])*1.2, 101)
                plt.plot(np.abs(correlations_pi[i,:]), marker='o', linestyle='', color=color, ms=8, label=f'J_parallel = {J_parallel:.2f}')
                plt.plot(fit_points, exp_fit(fit_points, *popt), color=color)

                # debug
                # plt.plot(np.abs(correlations_pi[i,:]), marker='o', linestyle='', color=color, ms=8, label=f'data')
                # plt.plot(fit_points, exp_fit(fit_points, *popt), color=color, label='fit')


                # plt.xlabel('rung index')
                # plt.ylabel('current correlation')
                # plt.title('Current Correlations')
                # plt.title(f'Current correlations for J_parallel/J_perp = {J_parallel/J_perp:.2f}')
                # plt.show()
            
            finally:
                coherence_length = popt[1]
                coherence_lengths[i] = coherence_length
                print(f'coherence length: {coherence_length}')
            
        plt.xlabel('rung index')
        plt.ylabel('current correlation')
        plt.title('Current Correlations')
        # plt.legend()
        plt.show()




        for i in range(correlations_0.shape[0]):
            # Map index i to a color from the matplotlib tableau color scheme
            color_1 = mpl.colormaps['bone'](i / correlations_0.shape[0])
            color_2 = mpl.colormaps['autumn'](i / correlations_0.shape[0])
            plt.plot(correlations_0[i,:], marker='o', linestyle=':', color=color_1, ms=8, label=f'J_parallel = {J_parallels[i]:.2f}')
            # plt.plot(np.abs(correlations_pi[i,:]), marker='o', linestyle='', color=color_2, ms=8, label=f'J_parallel = {J_parallels[i]:.2f}')
            plt.plot(np.abs(correlations_pi[i,:]), marker='x', linestyle=':', color=color_2, ms=8)

        plt.xlabel('rung index')
        plt.ylabel('current correlation')
        plt.title('Current Correlations')
        # plt.legend()
        plt.show()

        plt.plot(J_parallels, coherence_lengths, marker='o')
        plt.xlabel('J_parallel/J_perp')
        plt.ylabel('coherence length')
        plt.title('Coherence Length vs J_parallel/J_perp')
        plt.show()