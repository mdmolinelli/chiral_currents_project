import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QScrollArea, QWidget

from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from util.model_functions import lorentzian_fit




def get_data(filename, measurement='amp'):
    data1 = loadmat(filename)
    transAmpData1 = data1['transamp']
    specAmpData1 = data1['specamp']
    specPhaseData1 = data1['specphase']
    specFreqVector1 = data1['specfreq']
    volts1 = data1['voltage_vector']

    Ylist = specFreqVector1
    Xlist0 = volts1
    Xlist = Xlist0

    Xlist = np.asarray(Xlist)
    Ylist = np.asarray(Ylist) * 1e-9

    # Xlist = (Xlist - voltOffSet +VoltPerFlux*center)/VoltPerFlux

    X, Y = np.meshgrid(Xlist, Ylist)

    ### make copies of spec data
    phase = specPhaseData1.copy()
    amp = specAmpData1.copy()

    ### remove average for better plotting
    for i in range(0, len(phase[:, 1])):
        phase[i, :] = phase[i, :] - np.mean(phase[i, :])
        amp[i, :] = amp[i, :] - np.mean(amp[i, :])

    if measurement == 'amp':
        Z = amp.copy()
    elif measurement == 'phase':
        Z = phase.copy()
    Z = np.asarray(Z)
    Z = np.transpose(Z)

    return (X, Y, Z)


def plot_spec_data(figure, voltage_data, frequency_data, transmission_data, qubit_name=None, fit_voltages=None,
                   fit_frequencies=None, middle_frequency_index=None, separator_slope=None, vmin=-2, vmax=10):
    # Clear the existing figure
    figure.clf()

    # Create a new axis
    ax = figure.add_subplot(111)

    # To store axis limits
    voltage_min_all = float('inf')
    voltage_max_all = float('-inf')
    frequency_min_all = float('inf')
    frequency_max_all = float('-inf')

    for i in range(len(transmission_data)):
        voltage_min = voltage_data[i][0, 0]
        voltage_max = voltage_data[i][0, -1]
        voltage_step = voltage_data[i][0, 1] - voltage_data[i][0, 0]

        frequency_min = frequency_data[i][0, 0]
        frequency_max = frequency_data[i][-1, 0]
        frequency_step = frequency_data[i][1, 0] - frequency_data[i][0, 0]

        extent = (voltage_min - voltage_step / 2,
                  voltage_max + voltage_step / 2,
                  frequency_min - frequency_step / 2,
                  frequency_max + frequency_step / 2)

        # Use the ax object instead of plt
        im = ax.imshow(transmission_data[i], interpolation='none', vmin=vmin, vmax=vmax, origin='lower',
                       cmap='summer', aspect='auto', extent=extent, alpha=0.7)

        # Update the overall min/max for voltage and frequency
        voltage_min_all = min(voltage_min_all, extent[0])
        voltage_max_all = max(voltage_max_all, extent[1])
        frequency_min_all = min(frequency_min_all, extent[2])
        frequency_max_all = max(frequency_max_all, extent[3])

    # Set the overall axis limits using ax
    ax.set_xlim(voltage_min_all, voltage_max_all)
    ax.set_ylim(frequency_min_all, frequency_max_all)

    if fit_frequencies is not None and fit_voltages is not None:
        ax.plot(fit_voltages, fit_frequencies, marker='o', linestyle='', color='red', ms=2)

    if middle_frequency_index is not None:


        if separator_slope is None or separator_slope == 0: 
            # horizontal line
            middle_frequency_index = max(0, middle_frequency_index) 
            middle_frequency_index = min(len(frequency_data[0])-1, middle_frequency_index)

            ax.axhline(frequency_data[0][middle_frequency_index, 0], color='purple', linestyle=':')
        else:
            # sloped line
            num_voltage_points =voltage_data[0].shape[1]
            # line is frequency_index = slope * (voltage_index - num_voltage_points//2) + middle_frequency_index
            frequency_index_1 = int(separator_slope * (-num_voltage_points/2) + middle_frequency_index)
            frequency_index_2 = int(separator_slope * (num_voltage_points/2) + middle_frequency_index)

            frequency_index_1 = max(0, frequency_index_1)
            frequency_index_2 = min(frequency_data[0].shape[0]-1, frequency_index_2)

            voltage_1 = voltage_data[0][0, 0]
            voltage_2 = voltage_data[0][0, -1]

            frequency_1 = frequency_data[0][frequency_index_1, 0]
            frequency_2 = frequency_data[0][frequency_index_2, 0]

            ax.plot([voltage_1, voltage_2], [frequency_1, frequency_2], color='purple', linestyle=':')

    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Frequency (MHz)')
    cbar = figure.colorbar(im, ax=ax)
    cbar.set_label('Transmission (a.u.)')

    title = 'Qubit frequency vs voltage'
    if qubit_name is not None:
        title = f'{qubit_name} frequency vs voltage'
    ax.set_title(title)

def get_center_frequencies(voltage_data, frequency_data, transmission_data, start_index=5, frequency_index_span=100,
                           plot_fits=False):
    '''
    :param frequency_index_span: number of points around peak to try fit
    '''

    #     center_frequencies = np.zeros(voltage_data.shape[1])
    #     center_frequency_errors = np.zeros(voltage_data.shape[0])
    center_frequencies = []
    center_frequency_errors = []
    voltage_points_with_fit = []

    voltages = voltage_data[0, :]

    if isinstance(start_index, int):
        start_indices = [start_index] * len(voltages)
    else:
        start_indices = list(start_index)

    for i in range(len(voltages)):

        frequencies = frequency_data[:, 0]
        # find peak
        if i >= len(start_indices):
            start_index = start_indices[-1]
        else:
            start_index = start_indices[i]
        row = transmission_data[start_index:, i]

        peak_index = np.argmax(row) + start_index
        center_frequency_guess = frequencies[peak_index]

        # fit to lorentzian
        # restrict fit in range span around peak

        restricted_frequencies = frequencies[max(peak_index - frequency_index_span // 2, 0):min(
            peak_index + frequency_index_span // 2, len(frequencies))]
        restricted_row = transmission_data[
                         max(peak_index - frequency_index_span // 2, 0):min(peak_index + frequency_index_span // 2,
                                                                            len(frequencies)), i]

        # apply savgol filter

        filtered_row = savgol_filter(restricted_row, 7, 1)

        bounds = ([restricted_frequencies[0], 0, 0, -np.inf], [restricted_frequencies[-1], np.inf, np.inf, np.inf])
        initial_guess = [center_frequency_guess, 0.001, 0.0001, 0]
        try:
            popt, pcov = curve_fit(lorentzian_fit, restricted_frequencies, filtered_row, p0=initial_guess,
                                   bounds=bounds)
        except:
            # # if it fails, plot the data it was trying to fit
            # plt.plot(restricted_frequencies, filtered_row, linestyle='', marker='o', label='data')
            # plt.plot(restricted_frequencies, lorentzian_fit(restricted_frequencies, *initial_guess), label='guess')
            # plt.xlabel('Frequency (MHz)')
            # plt.title(f'Lorentzian fit for index {i}')
            # plt.axvline(center_frequency_guess, color='red')
            # plt.legend()
            # plt.show()

            # print('Couldn\'t get a fit')

            # use max as the center frequency
            center_frequencies.append(restricted_frequencies[np.argmax(filtered_row)])
            voltage_points_with_fit.append(voltages[i])
            center_frequency_errors.append(frequencies[-1] - frequencies[0])


        else:
            center_frequencies.append(popt[0])

            perr = np.sqrt(np.diag(pcov))
            center_frequency_errors.append(perr[0])

            voltage_points_with_fit.append(voltages[i])

            if plot_fits:
                plt.plot(frequencies[start_index:], row, linestyle='', marker='o', label='data')

                fit_frequencies = np.linspace(frequencies[start_index], frequencies[-1], 1000)
                plt.plot(fit_frequencies, lorentzian_fit(fit_frequencies, *popt), label='fit')
                plt.axvline(center_frequency_guess, color='red')
                plt.legend()

                plt.xlabel('Frequency (MHz)')
                plt.title(f'Lorentzian fit for index {i}')
                plt.show()

    return voltage_points_with_fit, center_frequencies, center_frequency_errors


def get_avoided_crossing_frequencies(voltage_data, frequency_data, transmission_data, voltage_start_index=0, middle_frequency_index=None, separator_slope=None, start_index=5, frequency_index_span=100, plot_fits=False):
    '''
    :param frequency_index_span: number of points around peak to try fit
    '''
    
    voltages = voltage_data[0,:]
    frequencies = frequency_data[:,0]

    center_frequencies_left = []
    center_frequency_errors_left = []
    voltages_with_fit_left = []
    
    center_frequencies_right = []
    center_frequency_errors_right = []
    voltages_with_fit_right = []
    
    if middle_frequency_index is None:
        middle_frequency_index = transmission_data.shape[0]//2

    if separator_slope is None:
        separator_slope = 0
        
    
    if isinstance(start_index, int):
        start_indices = [start_index] * transmission_data.shape[1]
    else:
        start_indices = list(start_index)

    if plot_fits:
        num_plots_per_window = 5
        axes = []
        for i in range(transmission_data.shape[1]//num_plots_per_window+1):
            fig, axes_i = plt.subplots(nrows=num_plots_per_window, ncols=1, figsize=(10, 5 * transmission_data.shape[1]//num_plots_per_window))
            axes.append(axes_i)
    
    for i in range(voltage_start_index, transmission_data.shape[1]):


        # find middle_frequency_index based on the separator slope
        if separator_slope is None or separator_slope == 0:
            middle_frequency_index_i = max(0, middle_frequency_index)
            middle_frequency_index_i = min(len(frequencies) - 1, middle_frequency_index)
        else:
            num_voltage_points = len(voltages)
            # line is frequency_index = slope * (voltage_index - num_voltage_points//2) + middle_frequency_index
            middle_frequency_index_i = int(separator_slope * (i - num_voltage_points / 2) + middle_frequency_index)

        # do the following twice, one for each half of the avoided crossing
        for j in range(2):
        
            # find peak
            if i >= len(start_indices):
                start_index = start_indices[-1]
            else:
                start_index = start_indices[i]
                
            if j == 0:
                # left half
                transmission_row = transmission_data[start_index:middle_frequency_index_i,i]
                frequency_row = frequencies[start_index:middle_frequency_index_i]

            else:
                transmission_row = transmission_data[middle_frequency_index_i:,i]
                frequency_row = frequencies[middle_frequency_index_i:]
            
            peak_index = np.argmax(transmission_row)
            peak_index = min(peak_index, len(frequencies) - 1)
            center_frequency_guess = frequency_row[peak_index]

            restricted_frequencies = frequency_row[max(peak_index - frequency_index_span//2, 0):min(peak_index + frequency_index_span//2, len(frequency_row))]
            restricted_row = transmission_row[max(peak_index - frequency_index_span//2, 0):min(peak_index + frequency_index_span//2, len(transmission_row))]

            
            # fit to lorentzian
            # restrict fit in range span around peak


            # apply savgol filter

            savgol_window = max(7, len(restricted_row)//2)
            filtered_row = savgol_filter(restricted_row, savgol_window, 1)

            bounds = ([restricted_frequencies[0], 0, 0, -np.inf], [restricted_frequencies[-1], np.inf, np.inf, np.inf])
            initial_guess = [center_frequency_guess, 0.001, 0.0001, 0]
            try:
                popt = None
                popt, pcov = curve_fit(lorentzian_fit, restricted_frequencies, filtered_row, p0=initial_guess, bounds=bounds)
            except:
                # if it fails, plot the data it was trying to fit
                # plt.plot(restricted_frequencies, filtered_row, linestyle='', marker='o', label='data')
                # plt.plot(restricted_frequencies, lorentzian_fit(restricted_frequencies, *initial_guess), label='guess')
                # plt.xlabel('Frequency (MHz)')
                # plt.title(f'Lorentzian fit for index {i}')
                # plt.axvline(center_frequency_guess, color='red', linestyle=':')
                # plt.legend()
                # plt.show()

                # print('Couldn\'t get a fit')

                # use max as the center frequency
                if j == 0:
                    center_frequencies_left.append(restricted_frequencies[np.argmax(filtered_row)])
                    voltages_with_fit_left.append(voltages[i])
                    center_frequency_errors_left.append(frequencies[-1] - frequencies[0])
                else:
                    center_frequencies_right.append(restricted_frequencies[np.argmax(filtered_row)])
                    voltages_with_fit_right.append(voltages[i])
                    center_frequency_errors_right.append(frequencies[-1] - frequencies[0])


            else:
                perr = np.sqrt(np.diag(pcov))

                if j == 0:
                    center_frequencies_left.append(popt[0])
                    center_frequency_errors_left.append(perr[0])
                    voltages_with_fit_left.append(voltages[i])
                else:
                    center_frequencies_right.append(popt[0])
                    center_frequency_errors_right.append(perr[0])
                    voltages_with_fit_right.append(voltages[i])

            if plot_fits:
                ax = axes[i // num_plots_per_window]
                ax = axes[i // num_plots_per_window][i % num_plots_per_window]    
                ax.plot(frequencies, transmission_data[:,i], linestyle='', marker='o', color='blue', label='data')

                fit_frequencies = np.linspace(frequencies[start_index], frequencies[-1], 1000)

                if popt is not None:
                    ax.plot(fit_frequencies, lorentzian_fit(fit_frequencies, *popt), color='green', label='fit')
                ax.axvline(center_frequency_guess, color='red', linestyle=':')
                
        if plot_fits:
            ax.axvline(frequencies[middle_frequency_index_i], color='purple', linestyle=':', label='middle cutoff frequency')
            # ax.legend()

            ax.set_xlabel('Frequency (MHz)')
            ax.set_title(f'Lorentzian fit for index {i}')

            if i % num_plots_per_window == 0:
                plt.tight_layout()
                plt.show()
            
    if plot_fits:
        # plt.tight_layout()
        # scrollable_window = ScrollableWindow(fig)
        plt.show()
            
    return [voltages_with_fit_left, voltages_with_fit_right], [center_frequencies_left, center_frequencies_right], [center_frequency_errors_left, center_frequency_errors_right]


# class ScrollableWindow(QMainWindow):
#     def __init__(self, fig):
#         self.qapp = QApplication.instance()
#         if not self.qapp:
#             self.qapp = QApplication([])

#         QMainWindow.__init__(self)
#         self.widget = QWidget()
#         self.setCentralWidget(self.widget)
#         self.widget.setLayout(QVBoxLayout())
#         self.widget.layout().setContentsMargins(0, 0, 0, 0)
#         self.widget.layout().setSpacing(0)

#         self.scroll = QScrollArea(self)
#         self.widget.layout().addWidget(self.scroll)
#         self.scroll.setWidgetResizable(True)

#         self.canvas = FigureCanvas(fig)
#         self.canvas.draw()
#         self.scroll.setWidget(self.canvas)

#         self.setWindowTitle("Scrollable Plot")
#         self.show()
#         self.qapp.exec_()