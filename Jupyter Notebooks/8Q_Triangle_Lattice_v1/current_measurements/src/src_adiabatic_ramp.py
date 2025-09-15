'''
This file defines the adiabatic ramp measurement class to measure populations of qubits after an adiabatic ramp.
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np

class AdiabaticRampGainMeasurement:
    def __init__(self, filename):
        self.filename = filename

        self.population = None
        self.population_corrected = None
        self.ff_gains = None

        self.readout_qubits = None

    def get_population(self):
        if self.population is None:
            self.acquire_data()
        return self.population
    
    def get_population_corrected(self):
        if self.population_corrected is None:
            self.acquire_data()
        return self.population_corrected
    
    def get_ff_gains(self):
        if self.ff_gains is None:
            self.acquire_data()
        return self.ff_gains
    
    def get_readout_qubits(self):
        if self.readout_qubits is None:
            self.acquire_data()
        return self.readout_qubits

    def acquire_data(self):
        # Placeholder for acquiring data logic
        self.population, self.population_corrected, self.ff_gains, self.readout_qubits = acquire_data(self.filename)

    def plot_populations(self, corrected=False, both=False):
        
        population = self.get_population()
        population_corrected = self.get_population_corrected()

        ff_gains = self.get_ff_gains()
        readout_qubits = self.get_readout_qubits()

        fig, ax = plt.subplots()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for i in range(len(population)):
            if both or not corrected:
                ax.plot(ff_gains, population[i], label=f'Qubit {readout_qubits[i]}', color=colors[i % len(colors)])
            if both:
                ax.plot(ff_gains, population_corrected[i], label=f'Qubit {readout_qubits[i]} Corrected', linestyle='--', color=colors[i % len(colors)])
            elif corrected:
                ax.plot(ff_gains, population_corrected[i], label=f'Qubit {readout_qubits[i]}', color=colors[i % len(colors)])

        ax.set_title('Populations after Adiabatic Ramp')
        ax.set_xlabel('Ramp Gain')
        ax.set_ylabel('Population')
        ax.legend()
        plt.show()

    def plot_population_bar_plot(self, corrected=False, gain_index=0, title=None):
        population = self.get_population()
        population_corrected = self.get_population_corrected()
        readout_qubits = self.get_readout_qubits()

        if corrected:
            data = population_corrected[:, gain_index]
        else:
            data = population[:, gain_index]

        fig, ax = plt.subplots()
        ax.bar(readout_qubits, data)

        ax.set_xticks(readout_qubits)
        ax.set_xticklabels([f'Q{i+1}' for i in readout_qubits])

        ax.set_ylim(0, 1)
        
        ax.set_ylabel('Population')

        if title is None:
            title = 'Population Bar Plot'
        ax.set_title(title)

        plt.show()
        





def acquire_data(filepath):
    with h5py.File(filepath, "r") as f:
        
        # for i in f:
            # print(f'{i}: {f[i][()]}')
            # print(i)

        population = f['population'][()]
        population_corrected = f['population_corrected'][()]
        ff_gains = f['Gain_Expt'][()]

        readout_list = [int(i) for i in f['readout_list'][()]]

    return population, population_corrected, ff_gains, readout_list


def generate_ramp_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampGainCalibration\RampGainCalibration_{}\RampGainCalibration_{}_{}_data.h5'.format(date_code, date_code, time_code)
