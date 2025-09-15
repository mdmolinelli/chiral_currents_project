import numpy as np
from scipy.optimize import curve_fit, root_scalar, minimize

########################################################################################
#      Import Tunable Transmon Parameters
########################################################################################

def frequency_model_fit(x, x0, a, b, c, d):
    return np.sqrt(np.abs(a)*np.sqrt(np.power(np.cos(b*(x-x0)),2) + (d**2)*np.power(np.sin(b*(x-x0)),2))) - c

def coupler_resonator_fit(x, x0, a, b, c, d, β, Ω, M):
     freq = np.sqrt(np.abs(a)*np.sqrt(np.power(np.cos(b*(x-x0)),2) + (d**2)*np.power(np.sin(b*(x-x0)),2))) - c
     resonator = Ω + M * x
     return resonator + β**2*freq / (resonator - freq)



def create_qubit_function(_popt):
        x0, a, b, c, d = _popt
        return lambda x: frequency_model_fit((np.pi*x)/b + x0, *_popt)

def create_qubit_inverse_function(qubit_function):
        def find_root(f, __qubit_function):
            bracket = (0, 0.5)

            if isinstance(f, (list, np.ndarray)):
                fluxes = np.empty(len(f))
                for i in range(len(f)):
                    root_function = lambda flux: __qubit_function(flux) - f[i]
                    result = root_scalar(root_function, bracket=bracket)
                    fluxes[i] = result.root
                return fluxes
            elif isinstance(f, (int, float)):
                root_function = lambda flux: __qubit_function(flux) - f
                result = root_scalar(root_function, bracket=bracket)
                return result.root
        return lambda f: find_root(f, qubit_function)

def create_qubit_flux_to_voltage(_popt):
        x0, a, b, c, d = _popt
        return lambda x: (np.pi*x)/b + x0
    
def create_qubit_voltage_to_flux(_popt):
        x0, a, b, c, d = _popt
        return lambda x: b*(x - x0)/np.pi

class Transmon:
    def __init__(self, transmon_popt, name='Qubit'):
        '''
        transmon_popt: [x0, a, b, c, d]'''
        x0, a, b, c, d = transmon_popt
        self.name = name
        self.transmon_popt = transmon_popt
        self.flux_quantum_voltage = np.pi/b
        self.voltage_offset       = x0

        self.freq = create_qubit_function(self.transmon_popt)     # flux -> freq
        self.flux = create_qubit_inverse_function(self.freq) # freq -> flux
        self.V_to_freq = lambda V: frequency_model_fit(V, *self.transmon_popt)
        self.flux_to_V = create_qubit_flux_to_voltage(self.transmon_popt)
        self.V_to_flux = create_qubit_voltage_to_flux(self.transmon_popt)