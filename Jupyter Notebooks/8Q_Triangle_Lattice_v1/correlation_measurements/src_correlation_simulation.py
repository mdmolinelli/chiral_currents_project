import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

class CorrelationSimulation:
    def __init__(self, num_levels, initial_state, J, U, detuning, T1=None, T2=None, times=None):
        self.num_levels = num_levels
        
        self.initial_state = initial_state
        self.detuning = detuning
        self.J = J
        self.U = U
        self.times = times

        if T1 is None:
            self.gamma_1 = 0
        else:
            self.gamma_1 = 2*np.pi/T1

        if T2 is None:
            self.gamma_2 = 0
        else:
            self.gamma_2 = 2*np.pi/T2

        self.gamma_phi = self.gamma_2 - self.gamma_1/2

        self.initialize_operators()
        
        self.result = None

        if not times is None:
            self.hamiltonian = generate_Hamiltonian(self.annihilation_operators, J, detuning, U)

        self.states = [self.initial_state]
        

        self.population_average = None
        self.standard_deviation = None
        self.covariance = None
        self.correlation = None

    def get_result(self):
        if self.result is None:
            self.run_simulation()
        return self.result
    
    def get_population_average(self):
        if self.population_average is None:
            self.population_average = np.zeros((len(self.number_operators), len(self.states)))
            for i in range(len(self.states)):
                self.population_average[:, i] = [qt.expect(op, self.states[i]) for op in self.number_operators]
        return self.population_average
    
    def get_standard_deviation(self):
        if self.standard_deviation is None:
            self.standard_deviation = np.zeros((len(self.number_operators), len(self.states)))
            for i in range(len(self.states)):
                self.standard_deviation[:, i] = np.sqrt([qt.expect(op*op, self.states[i]) - qt.expect(op, self.states[i])**2 for op in self.number_operators])
        return self.standard_deviation
    
    def get_covariance(self):
        if self.covariance is None:
            n1, n2 = self.number_operators
            population_average = self.get_population_average()
            self.covariance = np.zeros(len(self.states))
            for i in range(len(self.states)):
                self.covariance[i] = qt.expect(n1 * n2, self.states[i]) - population_average[0,i] * population_average[1,i]
        return self.covariance

    def run_simulation(self, psi0=None):

        c_ops = [np.sqrt(self.gamma_1) * op for op in self.annihilation_operators]
        c_ops += [np.sqrt(self.gamma_phi) * op.dag() * op for op in self.annihilation_operators]

        if psi0 is None:
            psi0 = self.initial_state
        self.result = qt.mesolve(self.hamiltonian, psi0, self.times, c_ops=c_ops)
        self.states = self.result.states

    def initialize_operators(self):

        a = qt.destroy(self.num_levels)
        a_1 = qt.tensor(a, qt.qeye(self.num_levels))
        a_2 = qt.tensor(qt.qeye(self.num_levels), a)
        self.annihilation_operators = (a_1, a_2)

        self.number_operators = [a.dag()*a for a in self.annihilation_operators]

def generate_Hamiltonian(annihilation_operators, J, detuning, U):
    # Create the Hamiltonian matrix using the provided parameters
    H = 0

    a_1, a_2 = annihilation_operators

    H = detuning * a_2.dag() * a_2
    H += J * (a_1.dag() * a_2 + a_2.dag() * a_1)
    H += U*(a_1.dag()*a_1*(a_1.dag()*a_1 - 1) + a_2.dag()*a_2*(a_2.dag()*a_2 - 1))

    return H