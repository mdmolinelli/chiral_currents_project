import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import qutip as qt

class QubitSwapTraceMeasurement:
    
    def __init__(self, readout_qubit, initial_state, coupling_sign, swap_filename, singleshot_filename, **kwargs):
        '''
        Measurement class for 3Q swaps
        :param readout_qubit: qubit whose population will be readout
        :param initial_state: Initial state prepared before swaps, either 'plus' or 'minus'
        :param coupling_sign: Sign of coupling strength between Q1 and Q3, either 'positive' or 'negative'
        :param singleshot_filename: complete filename of singleshot calibration data
        :param swap_filename: complete filename of swap data
        :param kwargs: used to add meta data for measurement object, such as what voltage/frequency the qubit/coupler was at
        
        '''
        
        self.readout_qubit = readout_qubit
        self.initial_state = initial_state
        self.coupling_sign = coupling_sign
        
        self.swap_filename = swap_filename
        self.singleshot_filename = singleshot_filename
        
        self.times = None
        self.populations = None
        
        self.times = None
        self.populations = None
        self.calibrated_populations = None
        self.simulated_populations = None
        
        self.i_g = None
        self.q_g = None
        self.i_e = None
        self.q_e = None
        
        self.angle = None
        self.threshold = None
        
        self.confusion_matrix = None
        self.confusion_matrix_inverse = None
        
        self.simulation_fit_parameters = None
        
        self.meta_data_dict = kwargs
        
    def get_times(self):
        if self.times is None:
            self.acquire_swap_data()
        return self.times
    
    def get_populations(self):
        if self.populations is None:
            self.acquire_swap_data()
        return self.populations
    
    def get_calibrated_populations(self):
        if self.calibrated_populations is None:
            self.generate_calibrated_populations()
        return self.calibrated_populations
    
    def get_simulated_populations(self, **kwargs):
        if self.simulated_populations is None:
            self.generate_simulated_populations(**kwargs)
        return self.simulated_populations
    
    def get_i_g(self):
        if self.i_g is None:
            self.acquire_singleshot_data()
        return self.i_g
    
    def get_q_g(self):
        if self.q_g is None:
            self.acquire_singleshot_data()
        return self.q_g
    
    def get_i_e(self):
        if self.i_e is None:
            self.acquire_singleshot_data()
        return self.i_e
    
    def get_q_e(self):
        if self.q_e is None:
            self.acquire_singleshot_data()
        return self.q_e
    
    def get_angle(self):
        if self.angle is None:
            self.acquire_singleshot_data()
        return self.angle
    
    def get_threshold(self):
        if self.threshold is None:
            self.acquire_singleshot_data()
        return self.threshold
    
    def get_confusion_matrix(self):
        if self.confusion_matrix is None:
            self.generate_confusion_matrix()
        return self.confusion_matrix
    
    def get_confusion_matrix_inverse(self):
        if self.confusion_matrix_inverse is None:
            self.generate_confusion_matrix()
        return self.confusion_matrix_inverse
    
    def get_simulation_fit_parameters(self, **kwargs):
        if self.simulation_fit_parameters is None:
            self.find_simulation_fit(**kwargs)
        return self.simulation_fit_parameters
        
    def acquire_swap_data(self):
        self.times, self.populations = acquire_swap_data(self.swap_filename)
        
    def acquire_singleshot_data(self):
        self.i_g, self.q_g, self.i_e, self.q_e, self.angle, self.threshold = acquire_singleshot_data(self.singleshot_filename, self.readout_qubit)
        
    def generate_confusion_matrix(self):
        
        i_g = self.get_i_g()
        q_g = self.get_q_g()
        i_e = self.get_i_e()
        q_e = self.get_q_e()
        
        angle = self.get_angle()
        threshold = self.get_threshold()
        
        i_g_new = i_g * np.cos(angle) - q_g * np.sin(angle)
        q_g_new = i_g * np.sin(angle) + q_g * np.cos(angle)
        i_e_new = i_e * np.cos(angle) - q_e * np.sin(angle)
        q_e_new = i_e * np.sin(angle) + q_e * np.cos(angle)
        
        # number of shots where g is measured after preparing g
        num_gg = sum(val < threshold for val in i_g_new)
        
        # number of shots where g is measured after preparing e
        num_ge = sum(val < threshold for val in i_e_new)
        
        # number of shots where e is measured after preparing g
        num_eg = sum(val >= threshold for val in i_g_new)
        
        # number of shots where e is measured after preparing e
        num_ee = sum(val >= threshold for val in i_e_new)
        
        # convert numbers to probabilities by dividing by total number of shots for g and e
        p_gg = num_gg/len(i_g)
        p_ge = num_ge/len(i_e)
        p_eg = num_eg/len(i_g)
        p_ee = num_ee/len(i_e)
        
        self.confusion_matrix = np.array([[p_gg, p_ge], [p_eg, p_ee]])
        self.confusion_matrix_inverse = np.linalg.inv(self.confusion_matrix)
        
    def generate_calibrated_populations(self):
        '''
        Calibrate populations by accounting for readout error using the inverse confusion matrix
        '''

        confusion_matrix_inverse = self.get_confusion_matrix_inverse()
        populations = self.get_populations()
        
        calibrated_populations = np.zeros(populations.shape)
        
        for i in range(len(populations)):
            population_vector = np.array([1 - populations[i], populations[i]])
            calibrated_population_vector = confusion_matrix_inverse @ population_vector
            
            calibrated_populations[i] = calibrated_population_vector[1]

        self.calibrated_populations = calibrated_populations
        
    def generate_simulated_populations(self, **kwargs):
        
        readout_qubits = ['Q1', 'Q2', 'Q3']
        readout_index = readout_qubits.index(self.readout_qubit)
        
        if not 'initial_state' in kwargs:
            kwargs['initial_state'] = self.initial_state
            
        self.simulated_populations = run_simulation(self.get_times(), readout_index=readout_index, **kwargs)
     
    def cost_function(self, x, times, populations, readout_index, **kwargs):
        g_12, g_23, g_13 = x
        kwargs['g_12'] = g_12
        kwargs['g_23'] = g_23
        kwargs['g_13'] = g_13

        simulated_populations = run_simulation(times, readout_index=readout_index, **kwargs)

        return np.sum(np.power(simulated_populations - populations, 2))
        
    def find_simulation_fit(self, **kwargs):
        
        
        times = self.get_times()
        populations = self.get_populations()
        
        readout_qubits = ['Q1', 'Q2', 'Q3']
        readout_index = readout_qubits.index(self.readout_qubit)
        
        if not 'initial_state' in kwargs:
            kwargs['initial_state'] = self.initial_state
        
        g_12 = kwargs['g_12']
        g_23 = kwargs['g_23']
        g_13 = kwargs['g_13']
        x0 = (g_12, g_23, g_13)
        bounds = ((1*2*np.pi, 20*2*np.pi), (1*2*np.pi, 20*2*np.pi), (1*2*np.pi, 20*2*np.pi))
        
        cost_function_lambda = lambda x: self.cost_function(x, times, populations, readout_index, **kwargs)
        
        optimize_result = minimize(cost_function_lambda, x0, bounds=bounds)
        
        print(cost_function_lambda(x0))
        print(cost_function_lambda(optimize_result.x))
        
        self.simulation_fit_parameters = optimize_result.x
        
    def plot_trace(self):
        
        times = self.get_times()
        populations = self.get_populations()
        
        plt.plot(times, populations, linestyle='', marker='o', ms=4)
        
        plt.xlabel('time (ns)')
        plt.ylabel('Population')
        
        plt.title(f'{self.readout_qubit} swap trace')
        plt.show()
        
    def plot_singleshot(self):
        
        i_g = self.get_i_g()
        q_g = self.get_q_g()
        i_e = self.get_i_e()
        q_e = self.get_q_e()
        
        angle = self.get_angle()
        threshold = self.get_threshold()
        
        i_g_new = i_g * np.cos(angle) - q_g * np.sin(angle)
        q_g_new = i_g * np.sin(angle) + q_g * np.cos(angle)
        i_e_new = i_e * np.cos(angle) - q_e * np.sin(angle)
        q_e_new = i_e * np.sin(angle) + q_e * np.cos(angle)
        
        fig, axes = plt.subplots(1, 2)
        
        axes[0].scatter(i_g, q_g, label='ground')
        axes[0].scatter(i_e, q_e, label='excited')
        
        axes[0].set_xlabel('I (a.u.)')
        axes[0].set_ylabel('Q (a.u.)')
        
        axes[0].set_title(f'{self.readout_qubit} single shot unrotated')
        
        axes[1].scatter(i_g_new, q_g_new, label='ground')
        axes[1].scatter(i_e_new, q_e_new, label='excited')
        
        axes[1].set_xlabel('I (a.u.)')
        axes[1].set_ylabel('Q (a.u.)')
        
        axes[1].set_title(f'{self.readout_qubit} single shot rotated')
        axes[1].axvline(threshold, color='black', linestyle=':', label='threshold')
        
        plt.show()
        
        
def acquire_swap_data(filepath):
    with h5py.File(filepath, "r") as f:
        
        time_units = 2.32515 / 16
        
        times = f['wait_times'][()]
        exp_data = f['Exp_values'][0,:]
        
    times *= time_units
    
    return times, exp_data

def acquire_singleshot_data(filepath, readout_qubit):
    
    readout_number = readout_qubit[1]
    
    with h5py.File(filepath, "r") as f:
        
        time_units = 2.32515 / 16
        
        i_g = f[f'i_g{readout_number}'][()]
        q_g = f[f'q_g{readout_number}'][()]
        
        i_e = f[f'i_e{readout_number}'][()]
        q_e = f[f'q_e{readout_number}'][()]
        
        angle = f['angle'][()][0]
        threshold = f['threshold'][()][0]
        
        return i_g, q_g, i_e, q_e, angle, threshold
    
def run_simulation(times, initial_state='plus', readout_index=None, **kwargs):
    '''
    Simulations swaps between 3 qubits after preparing the + or - two qubit state
    
    :param times: time points to evaluate simulation
    :initial_state: initial two qubit state between Q1 and Q2. Either 'plus' or 'minus'
    :readout_index: If provided, evaluate the populations for only that qubit, if None, evaluate all qubits
    :param omega_1: frequency in rotating frame of Q1 in MHz (*2PI)
    :param omega_2: frequency in rotating frame of Q2 in MHz (*2PI)
    :param omega_3: frequency in rotating frame of Q2 in MHz (*2PI)
    :param g_12: coupling bewteen Q1 and Q2 in MHz (*2PI)
    :param g_23: coupling bewteen Q2 and Q3 in MHz (*2PI)
    :param g_13: coupling bewteen Q1 and Q3 in MHz (*2PI)
    :param U: Anharmonicity of each qubit, can be given as list if they're different
    :param gamma_1: T1 decay rate of all qubits, can be given as list if they're different
    :param gamma_phi: Pure dephasing rate of all qubits, can be given as list if they're different
    :param num_levels: Number of levels for each qubit to include in the simulation
    '''
    
    if not 'omega_1' in kwargs:
        kwargs['omega_1'] = 0 # MHz
    omega_1 = kwargs['omega_1']
    
    if not 'omega_2' in kwargs:
        kwargs['omega_2'] = 0 # MHz
    omega_2 = kwargs['omega_2']
        
    if not 'omega_3' in kwargs:
        kwargs['omega_3'] = 0 # MHz
    omega_3 = kwargs['omega_3']
    
    
    ### check if coupler should be included
    if 'omega_c' in kwargs:
        omega_c = kwargs['omega_c']
        
        with_coupler = True
        num_qubits = 4
        
        # coupler default values
        if not 'g_1c' in kwargs:
            kwargs['g_1c'] = 100*2*np.pi # MHz
        g_1c = kwargs['g_1c']
        
        if not 'g_3c' in kwargs:
            kwargs['g_3c'] = 100*2*np.pi # MHz
        g_3c = kwargs['g_3c']
        
    else:
        with_coupler = False
        num_qubits = 3
    
    if not 'g_12' in kwargs:
        kwargs['g_12'] = 10*2*np.pi # MHz
    g_12 = kwargs['g_12']
    
    if not 'g_23' in kwargs:
        kwargs['g_23'] = 10*2*np.pi # MHz
    g_23 = kwargs['g_23']
    
    if not 'g_13' in kwargs:
        kwargs['g_13'] = 10*2*np.pi # MHz
    g_13 = kwargs['g_13']
    
    if not 'U' in kwargs:
        kwargs['U'] = 180 * 2 * np.pi # MHz
    U = kwargs['U']
    if isinstance(U, (list, tuple, np.ndarray)):
        _U = U
    elif isinstance(U, (int, float)):
        _U = [U]*num_qubits
        
        
       
    if not 'gamma_1' in kwargs:
        kwargs['gamma_1'] = 0.05 # MHz
    gamma_1 = kwargs['gamma_1']
    if isinstance(gamma_1, (list, tuple, np.ndarray)):
        _gamma_1 = gamma_1
    elif isinstance(gamma_1, float):
        _gamma_1 = [gamma_1]*num_qubits
    
    if not 'gamma_phi' in kwargs:
        kwargs['gamma_phi'] = 0.01 # MHz
    gamma_phi = kwargs['gamma_phi']
    if isinstance(gamma_phi, (list, tuple, np.ndarray)):
        _gamma_phi = gamma_phi
    elif isinstance(gamma_phi, (int, float)):
        _gamma_phi = [gamma_phi]*num_qubits
    
    if not 'num_levels' in kwargs:
        kwargs['num_levels'] = 3
    num_levels = kwargs['num_levels']
        
    a = qt.destroy(num_levels)
    
    if with_coupler:
        a1 = qt.tensor([a, qt.qeye(num_levels), qt.qeye(num_levels), qt.qeye(num_levels)])
        a2 = qt.tensor([qt.qeye(num_levels), a, qt.qeye(num_levels), qt.qeye(num_levels)])
        a3 = qt.tensor([qt.qeye(num_levels), qt.qeye(num_levels), a, qt.qeye(num_levels)])
        a_c = qt.tensor([qt.qeye(num_levels), qt.qeye(num_levels), qt.qeye(num_levels), a])
        
        U_1, U_2, U_3, U_c = _U
        gamma_1_1, gamma_1_2, gamma_1_3, gamma_1_c = _gamma_1
        gamma_phi_1, gamma_phi_2, gamma_phi_3, gamma_phi_c = _gamma_phi
        
        e_ops = [a1.dag()*a1, a2.dag()*a2, a3.dag()*a3, a_c.dag()*a_c]
        c_ops = [np.sqrt(gamma_1_1)*a1, np.sqrt(gamma_1_2)*a2, np.sqrt(gamma_1_3)*a2, np.sqrt(gamma_1_c)*a_c,
             np.sqrt(gamma_phi_1)*a1.dag()*a1, np.sqrt(gamma_phi_2)*a2.dag()*a2, np.sqrt(gamma_phi_3)*a3.dag()*a3, np.sqrt(gamma_phi_c)*a_c.dag()*a_c]
        
        H = generate_Hamiltonian_with_coupler([a1, a2, a3, a_c], omega_1, omega_2, omega_3, omega_c, g_12, g_23, g_13, g_1c, g_3c, U_1, U_2, U_3, U_c)
    else:
        a1 = qt.tensor([a, qt.qeye(num_levels), qt.qeye(num_levels)])
        a2 = qt.tensor([qt.qeye(num_levels), a, qt.qeye(num_levels)])
        a3 = qt.tensor([qt.qeye(num_levels), qt.qeye(num_levels), a])
        
        U_1, U_2, U_3 = _U
        gamma_1_1, gamma_1_2, gamma_1_3 = _gamma_1
        gamma_phi_1, gamma_phi_2, gamma_phi_3 = _gamma_phi
        
        e_ops = [a1.dag()*a1, a2.dag()*a2, a3.dag()*a3]
        c_ops = [np.sqrt(gamma_1_1)*a1, np.sqrt(gamma_1_2)*a2, np.sqrt(gamma_1_3)*a2,
             np.sqrt(gamma_phi_1)*a1.dag()*a1, np.sqrt(gamma_phi_2)*a2.dag()*a2, np.sqrt(gamma_phi_3)*a3.dag()*a3]
        
        H = generate_Hamiltonian([a1, a2, a3], omega_1, omega_2, omega_3, g_12, g_23, g_13, U_1, U_2, U_3)
    
    if isinstance(initial_state, str):
        if initial_state == 'plus':
            if with_coupler:
                psi0 = 1/np.sqrt(2)*(qt.basis([num_levels, num_levels, num_levels, num_levels], [1, 0, 0, 0]) + qt.basis([num_levels, num_levels, num_levels, num_levels], [0, 0, 1, 0]))
            else:
                psi0 = 1/np.sqrt(2)*(qt.basis([num_levels, num_levels, num_levels], [1, 0, 0]) + qt.basis([num_levels, num_levels, num_levels], [0, 0, 1]))
                
        elif initial_state == 'minus':
            if with_coupler:
                psi0 = 1/np.sqrt(2)*(qt.basis([num_levels, num_levels, num_levels, num_levels], [1, 0, 0, 0]) - qt.basis([num_levels, num_levels, num_levels, num_levels], [0, 0, 1, 0]))
            else:
                psi0 = 1/np.sqrt(2)*(qt.basis([num_levels, num_levels, num_levels], [1, 0, 0]) - qt.basis([num_levels, num_levels, num_levels], [0, 0, 1]))
    elif isinstance(initial_state, qt.core.qobj.Qobj):
        psi0 = initial_state
        
        
        
    if readout_index is not None and 0 <= readout_index < len(e_ops):
        e_ops = [e_ops[readout_index]]
    
    
    
    result = qt.mesolve(H, psi0, times/1e3, e_ops=e_ops, c_ops=c_ops)
    
    populations = np.zeros((len(e_ops), len(times)))
    for i in range(len(e_ops)):
        populations[i,:] = result.expect[i]
        
    if readout_index is not None:
        return populations[0]
    else:
        return populations
    
def generate_Hamiltonian(annihilation_operators, omega_1, omega_2, omega_3, g_12, g_23, g_13, U_1, U_2, U_3):
                  
    a1, a2, a3 = annihilation_operators
    
    H = omega_1 * a1.dag()*a1 + omega_2 * a2.dag()*a2 + omega_3 * a3.dag()*a3 
    H += g_12*(a1*a2.dag() + a2*a1.dag()) + g_23*(a3*a2.dag() + a2*a3.dag()) + g_13*(a1*a3.dag() + a3*a1.dag())
    H += U_1*a1.dag()*a1*(a1.dag()*a1 - 1) + U_2*a2.dag()*a2*(a2.dag()*a2 - 1) + U_3*a3.dag()*a3*(a3.dag()*a3 - 1)
    
    return H

def generate_Hamiltonian_with_coupler(annihilation_operators, omega_1, omega_2, omega_3, omega_c, g_12, g_23, g_13, g_1c, g_3c, U_1, U_2, U_3, U_c):
                  
    a1, a2, a3, a_c = annihilation_operators
    
    print(np.array([g_12, g_23, g_13, g_1c, g_3c])/2/np.pi)
    print(np.array([omega_1, omega_2, omega_3, omega_c])/2/np.pi)
    
    H = omega_1 * a1.dag()*a1 + omega_2 * a2.dag()*a2 + omega_3 * a3.dag()*a3 + omega_c * a_c.dag()*a_c 
    H += g_12*(a1*a2.dag() + a2*a1.dag()) + g_23*(a3*a2.dag() + a2*a3.dag()) + g_13*(a1*a3.dag() + a3*a1.dag())
    H += g_1c*(a1*a_c.dag() + a_c*a1.dag()) + g_3c*(a3*a_c.dag() + a_c*a3.dag())
    H += U_1*a1.dag()*a1*(a1.dag()*a1 - 1) + U_2*a2.dag()*a2*(a2.dag()*a2 - 1) + U_3*a3.dag()*a3*(a3.dag()*a3 - 1) + U_c*a_c.dag()*a_c*(a_c.dag()*a_c - 1)
    
    return H
        
def generate_swap_filename(year, month, day, hour, minute, second, adiabatic=False):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    if adiabatic:
        return r'V:\QSimMeasurements\Measurements\5QV2_Triangle_Lattice\AdiabaticRampOscillations\AdiabaticRampOscillations_{}\AdiabaticRampOscillations_{}_{}_data.h5'.format(date_code, date_code, time_code)
    else:
        return r'V:\QSimMeasurements\Measurements\5QV2_Triangle_Lattice\QubitOscillationsMUX\QubitOscillationsMUX_{}\QubitOscillationsMUX_{}_{}_data.h5'.format(date_code, date_code, time_code)
    
    
def generate_singleshot_filename(year, month, day, hour, minute, second, local=False):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    if local:
        return r'C:\Users\mattm\OneDrive\Desktop\Research\Measurements\5QV2_Triangle_Lattice\SingleShot\SingleShot_{}\SingleShot_{}_{}_data.h5'.format(date_code, date_code, time_code)
    else:
        return r'V:\QSimMeasurements\Measurements\5QV2_Triangle_Lattice\SingleShot\SingleShot_{}\SingleShot_{}_{}_data.h5'.format(date_code, date_code, time_code)
    

