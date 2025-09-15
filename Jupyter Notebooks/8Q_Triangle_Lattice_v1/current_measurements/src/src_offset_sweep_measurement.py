import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq

class OffsetSweepMeasurement:

    ns_per_sample = 2.32515*2 / 16 # tproc_V2

    def __init__(self, filename, ramp=False, readout_indices=None):
        self.filename = filename
        self.ramp = ramp

        if isinstance(readout_indices, int):
            readout_indices = [readout_indices]
        self.readout_indices = readout_indices

        self.times = None
        self.offset_times = None # in ns

        self.population = None
        self.population_corrected = None

        self.readout_qubits = None

        self.offset_phase = None
        self.offset_frequency = None
        self.offset_frequency_error = None


    def get_times(self):
        if self.times is None:
            self.acquire_data()
        return self.times

    def get_offset_times(self):
        if self.offset_times is None:
            self.acquire_data()
        return self.offset_times

    def get_population(self):
        if self.population is None:
            self.acquire_data()
        return self.population
    
    def get_population_corrected(self):
        if self.population_corrected is None:
            self.acquire_data()
        return self.population_corrected
    
    def get_readout_qubits(self):
        if self.readout_qubits is None:
            self.acquire_data()
        return self.readout_qubits
    
    def get_offset_phase(self):
        if self.offset_phase is None:
            self.calculate_offset_sin_fit()
        return self.offset_phase
    
    def get_offset_frequency(self):
        if self.offset_frequency is None:
            self.calculate_offset_sin_fit()
        return self.offset_frequency
    
    def calculate_offset_sin_fit(self, readout_index=None, time_slice_index=0, sin_start_index=None, sin_end_index=None, plot_sin_fit=False):

        if readout_index is None:
            if self.readout_indices is None:
                readout_index = 0
            else:
                readout_index = self.readout_indices[0]

        population = self.get_population_corrected()
        offset_times = self.get_offset_times()


        slice = population[readout_index, :, time_slice_index]

        frequency_guess = get_frequency_from_fft(slice, len(offset_times), offset_times[1] - offset_times[0])

        sin_initial_guess = [frequency_guess, 0, 0.8, 0.5, 0]

        sin_popt, sin_perr = get_sin_fit_parameters(offset_times, slice, guess=sin_initial_guess, start_index=sin_start_index, end_index=sin_end_index)

        self.offset_frequency = sin_popt[0] # GHz
        self.offset_frequency_error = sin_perr[0]

        self.offset_phase = 2*np.pi*self.offset_frequency * offset_times + sin_popt[1]

        if plot_sin_fit:
            plt.figure(figsize=(10, 5))
            plt.plot(offset_times, slice, label='Data', linestyle='', marker='o')

            fit_times = np.linspace(offset_times[0], offset_times[-1], 1001)

            plt.plot(fit_times, sin_fit(fit_times, *sin_initial_guess), label='Initial Guess', linestyle=':')
            plt.plot(fit_times, sin_fit(fit_times, *sin_popt), label='Sin Fit', linestyle='--')
            plt.xlabel('Offset Time (ns)')
            plt.ylabel('Population')
            plt.title(f'Sin Fit for Readout Qubit {self.readout_qubits[readout_index]}')
            plt.legend()
            plt.grid()
            plt.show()

    def acquire_data(self):
        self.times, self.offset_times,  self.population, self.population_corrected, self.readout_qubits = acquire_data_offset_sweep(self.filename, self.ramp)

    def plot_population(self, corrected=False, plot_vs_phase=False, readout_indices=None, subtitle=None):
        '''


        :param corrected: plot corrected population
            (Default: False)
        :param plot_vs_phase: plot y axis as phase instead of offset time
            (Default: False)
        :param readout_indices: plot only these readout indices, if None plot all
        '''

        if corrected:
            population = self.get_population_corrected()
        else:
            population = self.get_population()


        readout_qubits = self.get_readout_qubits()

        if readout_indices is None:
            readout_indices = self.readout_indices
            if readout_indices is None:
                readout_indices = list(range(population.shape[0]))
        else:
            if isinstance(readout_indices, int):
                readout_indices = [readout_indices]

        if plot_vs_phase:
            phase = self.get_offset_phase()

        

        num_indices = len(readout_indices)
        ncols = min(2, num_indices)
        nrows = (num_indices + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows))
        axs = np.array(axs).reshape(-1)  # Flatten in case axs is 2D


        x_axis = self.get_times()
        y_axis = self.get_offset_times()

        x_step = x_axis[1] - x_axis[0]
        y_step = y_axis[1] - y_axis[0]

        extent = (x_axis[0] - x_step / 2, x_axis[-1] + x_step / 2, y_axis[0] - y_step / 2, y_axis[-1] + y_step / 2)

        for idx, readout_idx in enumerate(readout_indices):
            im = axs[idx].imshow(population[readout_idx], aspect='auto', origin='lower', interpolation='none', extent=extent)

            if subtitle is None:
                subtitle = f'Q{readout_qubits[readout_idx]}'
            axs[idx].set_title(subtitle)

            plt.colorbar(im, ax=axs[idx], label='Population', pad=0.15)
            axs[idx].set_xlabel('Time (ns)')
            axs[idx].set_ylabel('Offset Time (ns)')

            if plot_vs_phase:
                # Add phase as the right y-axis labels
                ax2 = axs[idx].twinx()
                ax2.set_ylim(axs[idx].get_ylim())
                
                # Find integer multiples of pi within the phase range
                phase_min, phase_max = np.min(phase), np.max(phase)
                pi_min = int(np.ceil(phase_min / np.pi))
                pi_max = int(np.floor(phase_max / np.pi))
                
                # Create pi tick values and corresponding offset times
                pi_multiples = np.arange(pi_min, pi_max + 1)
                pi_phase_values = pi_multiples * np.pi
                
                # Find corresponding offset times for each pi multiple
                pi_offset_times = []
                pi_labels = []
                for pi_val in pi_phase_values:
                    # Find closest phase value and corresponding offset time
                    closest_idx = np.argmin(np.abs(phase - pi_val))
                    pi_offset_times.append(y_axis[closest_idx])
                    
                    # Create nice labels for pi multiples
                    if pi_multiples[len(pi_labels)] == 0:
                        pi_labels.append('0')
                    elif pi_multiples[len(pi_labels)] == 1:
                        pi_labels.append('π')
                    elif pi_multiples[len(pi_labels)] == -1:
                        pi_labels.append('-π')
                    else:
                        pi_labels.append(f'{pi_multiples[len(pi_labels)]}π')
                
                ax2.set_yticks(pi_offset_times)
                ax2.set_yticklabels(pi_labels)
                ax2.set_ylabel('Phase (rad)')

        # Hide any unused subplots
        for j in range(num_indices, nrows * ncols):
            axs[j].axis('off')

        plt.tight_layout()

    def plot_swap_trace(self, corrected=False, readout_indices=None, offset_phase_value=None, offset_index=None, ylim=None, title=None):

        offset_phase = self.get_offset_phase()


        if readout_indices is None:
            readout_indices = self.readout_indices
            if readout_indices is None:
                readout_indices = list(range(self.population.shape[0]))
        else:
            if isinstance(readout_indices, int):
                readout_indices = [readout_indices]

        if offset_phase_value is None:
            if offset_index is None:
                offset_index = 0
            offset_phase_value = offset_phase[offset_index]
        else:
            offset_index = np.argmin(np.abs(offset_phase - offset_phase_value))

        if corrected:
            population = self.get_population_corrected()
        else:
            population = self.get_population()

        swap_trace = population[:, offset_index, :]

        readout_qubits = self.get_readout_qubits()

        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, readout_idx in enumerate(readout_indices):
            ax.plot(self.get_times(), swap_trace[i,:], label=f'Qubit {readout_qubits[readout_idx]}', linestyle='', marker='o')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Population')

        if title is None:
            title = f'Qubit swaps with phase {offset_phase_value/np.pi:.2f} π'
        ax.set_title(title)        

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.legend()
        plt.show()

class OffsetGainSweepMeasurement:
    
    ns_per_sample = 2.32515*2 / 16 # tproc_V2

    def __init__(self, filename, ramp=False, readout_indices=None):
        self.filename = filename
        self.ramp = ramp

        if isinstance(readout_indices, int):
            readout_indices = [readout_indices]
        self.readout_indices = readout_indices

        self.times = None
        self.ff_gains = None

        self.population = None
        self.population_corrected = None

        self.readout_qubits = None

        self.offset_phase = None
        self.offset_time= None
        self.offset_time_error = None


    def get_times(self):
        if self.times is None:
            self.acquire_data()
        return self.times

    def get_ff_gains(self):
        if self.ff_gains is None:
            self.acquire_data()
        return self.ff_gains

    def get_population(self):
        if self.population is None:
            self.acquire_data()
        return self.population
    
    def get_population_corrected(self):
        if self.population_corrected is None:
            self.acquire_data()
        return self.population_corrected
    
    def get_readout_qubits(self):
        if self.readout_qubits is None:
            self.acquire_data()
        return self.readout_qubits
    
    def get_offset_phase(self):
        if self.offset_phase is None:
            self.calculate_offset_sin_fit()
        return self.offset_phase
    
    def get_offset_time(self):
        if self.offset_time is None:
            self.calculate_offset_sin_fit()
        return self.offset_time
    
    def calculate_offset_sin_fit(self, readout_index=None, time_slice_index=0, sin_start_index=None, sin_end_index=None, plot_sin_fit=False):

        if readout_index is None:
            if self.readout_indices is None:
                readout_index = 0
            else:
                readout_index = self.readout_indices[0]

        population = self.get_population_corrected()
        ff_gains = self.get_ff_gains()


        slice = population[readout_index, :, time_slice_index]

        time_guess = get_frequency_from_fft(slice, len(ff_gains), ff_gains[1] - ff_gains[0])

        sin_initial_guess = [time_guess, 0, 0.8, 0.5, 0]

        sin_popt, sin_perr = get_sin_fit_parameters(ff_gains, slice, guess=sin_initial_guess, start_index=sin_start_index, end_index=sin_end_index)

        self.offset_time = sin_popt[0] # ns
        self.offset_time_error = sin_perr[0]

        self.offset_phase = 2*np.pi*self.offset_time * ff_gains + sin_popt[1]

        if plot_sin_fit:
            plt.figure(figsize=(10, 5))
            plt.plot(ff_gains, slice, label='Data', linestyle='', marker='o')

            fit_gains = np.linspace(ff_gains[0], ff_gains[-1], 1001)

            plt.plot(fit_gains, sin_fit(fit_gains, *sin_initial_guess), label='Initial Guess', linestyle=':')
            plt.plot(fit_gains, sin_fit(fit_gains, *sin_popt), label='Sin Fit', linestyle='--')
            plt.xlabel('FF Gains (a.u.)')
            plt.ylabel('Population')
            plt.title(f'Sin Fit for Readout Qubit {self.readout_qubits[readout_index]}')
            plt.legend()
            plt.grid()
            plt.show()

    def acquire_data(self):
        self.times, self.ff_gains, self.population, self.population_corrected, self.readout_qubits = acquire_data_gain_sweep(self.filename, self.ramp)

    def plot_population(self, corrected=False, plot_vs_phase=False, readout_indices=None, subtitle=None):
        '''


        :param corrected: plot corrected population
            (Default: False)
        :param plot_vs_phase: plot y axis as phase instead of offset time
            (Default: False)
        :param readout_indices: plot only these readout indices, if None plot all
        '''

        if corrected:
            population = self.get_population_corrected()
        else:
            population = self.get_population()


        readout_qubits = self.get_readout_qubits()

        if readout_indices is None:
            readout_indices = self.readout_indices
            if readout_indices is None:
                readout_indices = list(range(population.shape[0]))
        else:
            if isinstance(readout_indices, int):
                readout_indices = [readout_indices]

        if plot_vs_phase:
            phase = self.get_offset_phase()

        

        num_indices = len(readout_indices)
        ncols = min(2, num_indices)
        nrows = (num_indices + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows))
        axs = np.array(axs).reshape(-1)  # Flatten in case axs is 2D


        x_axis = self.get_times()
        y_axis = self.get_ff_gains()

        x_step = x_axis[1] - x_axis[0]
        y_step = y_axis[1] - y_axis[0]

        extent = (x_axis[0] - x_step / 2, x_axis[-1] + x_step / 2, y_axis[0] - y_step / 2, y_axis[-1] + y_step / 2)

        for idx, readout_idx in enumerate(readout_indices):
            im = axs[idx].imshow(population[readout_idx], aspect='auto', origin='lower', interpolation='none', extent=extent)

            if subtitle is None:
                subtitle = f'Q{readout_qubits[readout_idx]}'
            axs[idx].set_title(subtitle)

            plt.colorbar(im, ax=axs[idx], label='Population', pad=0.15)
            axs[idx].set_xlabel('Time (ns)')
            axs[idx].set_ylabel('FF Gain (a.u.)')

            if plot_vs_phase:
                # Add phase as the right y-axis labels
                ax2 = axs[idx].twinx()
                ax2.set_ylim(axs[idx].get_ylim())
                
                # Find integer multiples of pi within the phase range
                phase_min, phase_max = np.min(phase), np.max(phase)
                pi_min = int(np.ceil(phase_min / np.pi))
                pi_max = int(np.floor(phase_max / np.pi))
                
                # Create pi tick values and corresponding offset times
                pi_multiples = np.arange(pi_min, pi_max + 1)
                pi_phase_values = pi_multiples * np.pi
                
                # Find corresponding offset times for each pi multiple
                pi_offset_times = []
                pi_labels = []
                for pi_val in pi_phase_values:
                    # Find closest phase value and corresponding offset time
                    closest_idx = np.argmin(np.abs(phase - pi_val))
                    pi_offset_times.append(y_axis[closest_idx])
                    
                    # Create nice labels for pi multiples
                    if pi_multiples[len(pi_labels)] == 0:
                        pi_labels.append('0')
                    elif pi_multiples[len(pi_labels)] == 1:
                        pi_labels.append('π')
                    elif pi_multiples[len(pi_labels)] == -1:
                        pi_labels.append('-π')
                    else:
                        pi_labels.append(f'{pi_multiples[len(pi_labels)]}π')
                
                ax2.set_yticks(pi_offset_times)
                ax2.set_yticklabels(pi_labels)
                ax2.set_ylabel('Phase (rad)')

        # Hide any unused subplots
        for j in range(num_indices, nrows * ncols):
            axs[j].axis('off')

        plt.tight_layout()

    def plot_swap_trace(self, corrected=False, readout_indices=None, offset_phase_value=None, offset_index=None, ylim=None, title=None):

        offset_phase = self.get_offset_phase()


        if readout_indices is None:
            readout_indices = self.readout_indices
            if readout_indices is None:
                readout_indices = list(range(self.population.shape[0]))
        else:
            if isinstance(readout_indices, int):
                readout_indices = [readout_indices]

        if offset_phase_value is None:
            if offset_index is None:
                offset_index = 0
            offset_phase_value = offset_phase[offset_index]
        else:
            offset_index = np.argmin(np.abs(offset_phase - offset_phase_value))

        if corrected:
            population = self.get_population_corrected()
        else:
            population = self.get_population()

        swap_trace = population[:, offset_index, :]

        readout_qubits = self.get_readout_qubits()

        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, readout_idx in enumerate(readout_indices):
            ax.plot(self.get_times(), swap_trace[i,:], label=f'Qubit {readout_qubits[readout_idx]}', linestyle='', marker='o')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Population')

        if title is None:
            title = f'Qubit swaps with phase {offset_phase_value/np.pi:.2f} π'
        ax.set_title(title)        

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.legend()
        plt.show()

def acquire_data_offset_sweep(filepath, ramp=False):
    with h5py.File(filepath, "r") as f:
        
        time_units = 2.32515 / 16 # tproc_V1
        time_units = 2.32515*2 / 16 # tproc_V2
        
        for i in f:
            # print(f'{i}: {f[i][()]}')
            print(i)

        if 'expt_samples' in f:
            times = f['expt_samples'][()]

        elif 'expt_samples2' in f:
            times = f['expt_samples2'][()]

        population = f['population'][()]
        population_corrected = f['population_corrected'][()]

        if 't_offset' in f:
            offset_times = f['t_offset'][()]
        elif 'intermediate_jump_samples' in f:
            offset_times = f['intermediate_jump_samples'][()]

        readout_qubits = [int(i) for i in f['readout_list'][()]]


    times *= time_units
    offset_times *= time_units

    return times, offset_times, population, population_corrected, readout_qubits

def acquire_data_gain_sweep(filepath, ramp=False):
    with h5py.File(filepath, "r") as f:
        
        time_units = 2.32515 / 16 # tproc_V1
        time_units = 2.32515*2 / 16 # tproc_V2
        
        for i in f:
            # print(f'{i}: {f[i][()]}')
            print(i)

        times = f['expt_samples'][()]

        population = f['population'][()]
        population_corrected = f['population_corrected'][()]

        ff_gains = f['intermediate_jump_gains'][()]

        readout_qubits = [int(i) for i in f['readout_list'][()]]


    times *= time_units

    return times, ff_gains, population, population_corrected, readout_qubits

    

def get_sin_fit_parameters(times, exp, guess=None, start_index=None, end_index=None):
    
    if start_index is None:
        start_index = 0
    
    if end_index is None:
        end_index = len(times)
    else:
        end_index = min(end_index, len(times))
        
    popt, pcov = curve_fit(sin_fit, times[start_index: end_index], exp[start_index: end_index], p0=guess, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))
    
#     print(fit_p)
#     print(fit_p[0] / 2, fit_err[0] / 2)
    return popt, perr

def sin_fit(t, f, phi0, A, B, gamma):
    return A * np.sin(2 * np.pi * f * t + phi0)*np.exp(-gamma*t) + B

def get_frequency_from_fft(exp, num_times, times_spacing, start_index=1, plot_spectra=False):
    exp_fft = rfft(exp)
    freqs_fft = rfftfreq(num_times, times_spacing) # GHz

    peak_index = np.argmax(np.abs(exp_fft[start_index:])) + start_index
    center_freq = freqs_fft[peak_index]
    
#     print(f'peak index: {peak_index}')
#     print(f'fft: {exp_fft}')
#     print(f'freqs: {freqs_fft}')

    if plot_spectra:
        plt.plot(freqs_fft, np.abs(exp_fft))
        plt.axvline(center_freq, linestyle='dashed', color='black')
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Amplitude')
        plt.show()
        
    return center_freq

def convert_samples_to_ns(samples):
    return samples * OffsetSweepMeasurement.ns_per_sample

def convert_ns_to_samples(ns):
    return ns / OffsetSweepMeasurement.ns_per_sample

def generate_offset_sweep_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\CurrentCalibration_OffsetSweep\CurrentCalibration_OffsetSweep_{}\CurrentCalibration_OffsetSweep_{}_{}_data.h5'.format(date_code, date_code, time_code)

def generate_ramp_offset_sweep_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampCurrentCalibration_OffsetSweep\RampCurrentCalibration_OffsetSweep_{}\RampCurrentCalibration_OffsetSweep_{}_{}_data.h5'.format(date_code, date_code, time_code)


def generate_ramp_double_jump_offset_sweep_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampDoubleJumpIntermediateLength\RampDoubleJumpIntermediateLength_{}\RampDoubleJumpIntermediateLength_{}_{}_data.h5'.format(date_code, date_code, time_code)


def generate_ramp_double_jump_gain_sweep_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampDoubleJumpGainR\RampDoubleJumpGainR_{}\RampDoubleJumpGainR_{}_{}_data.h5'.format(date_code, date_code, time_code)