import sys
from itertools import product
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QLineEdit, QPushButton, \
    QCheckBox, QGroupBox, QGridLayout, QFileDialog, QScrollArea, QSizePolicy, QComboBox, QButtonGroup, QRadioButton, \
    QTextEdit
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import os

from scipy.io import loadmat
from scipy.optimize import curve_fit, root_scalar
from scipy.signal import savgol_filter


class SpectroscopyMeasurementsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        tabs = QTabWidget()

        self.qubits_tab = SpectroscopySubTabQubit("Qubit Spectroscopy")
        self.couplers_tab = SpectroscopySubTabCoupler("Coupler Spectroscopy")

        tabs.addTab(self.qubits_tab, "Qubit Spectroscopy")
        tabs.addTab(self.couplers_tab, "Coupler Spectroscopy")

        layout.addWidget(tabs)
        self.setLayout(layout)


class SpectroscopySubTabQubit(QWidget):
    qubits = ['Q2', 'Q3', 'Q4']

    voltage_sweep = 'Voltage Sweep'
    flux_sweep = 'Flux Sweep'
    random_voltages = 'Random Voltages'

    spec_fit_section = 'spec_fit'
    calibration_section = 'calibration'

    def __init__(self, tab_name):
        super().__init__()
        self.tab_name = tab_name

        self.qubit_to_spec_canvas = {}

        # label data by qubit and section
        # i.e. self.qubit_to_voltage_data['Q1']['spec_fit']
        self.qubit_to_voltage_data = {qubit: {} for qubit in self.qubits}
        self.qubit_to_frequency_data = {qubit: {} for qubit in self.qubits}
        self.qubit_to_transmission_data = {qubit: {} for qubit in self.qubits}

        # self.qubit_to_frequencies_canvas = {}

        ### indices of qubit spectroscopy fits to ignore, do this manually for now
        self.qubit_to_ignore_indices = {}

        self.qubit_to_ignore_indices['Q2'] = [30, 31, (35, 40), 42, 43, 70, 75, 86, 100, 145, 174, 176, 179, 181, 184,
                                              187,
                                              189, 192, 194,
                                              197, 250, 252, 254, 255, 257, 258, 260, 262, 263, 265, 267, 268, 270, 272,
                                              273,
                                              275, 276,
                                              (437, 450)]

        self.qubit_to_ignore_indices['Q3'] = [118, 158, 250, 267, 364, (386, 394)]

        self.qubit_to_ignore_indices['Q4'] = [42, (151, 163), 166, 186, 221, 231, 256, 262, 270]

        # extracted voltages and frequencies
        self.qubit_to_voltages = {qubit: {} for qubit in self.qubits}
        self.qubit_to_frequencies = {qubit: {} for qubit in self.qubits}

        self.qubit_to_filtered_voltages = {}
        self.qubit_to_filtered_frequencies = {}

        # qubit function fit and inverse function
        self.qubit_to_fit_function_canvases = {}

        # fit initial guesses and bounds, do manually for now

        # initial guesses
        self.qubit_to_trianglemon_initial_guess = {}

        self.qubit_to_trianglemon_initial_guess['Q2'] = [-0.5, 30, 1.2, 0.2006, 0.6, 0.1]
        self.qubit_to_trianglemon_initial_guess['Q3'] = [-0.5, 30, 1.2, 0.2004, 0.6, 0.1]
        self.qubit_to_trianglemon_initial_guess['Q4'] = [-0.5, 30, 1.2, 0.2002, 0.6, 0.1]

        # bounds
        self.qubit_to_trianglemon_bounds = {}

        self.qubit_to_trianglemon_bounds['Q2'] = (
            (-np.inf, 0, 0, 0.2005, 0, 0), (np.inf, np.inf, np.inf, 0.2006, 1, np.inf))
        self.qubit_to_trianglemon_bounds['Q3'] = (
            (-np.inf, 0, 0, 0.2004, 0, 0), (np.inf, np.inf, np.inf, 0.2005, 1, np.inf))
        self.qubit_to_trianglemon_bounds['Q4'] = (
            (-np.inf, 0, 0, 0.2002, 0, 0), (np.inf, np.inf, np.inf, 0.2003, 1, np.inf))

        # qubit function and inverse
        self.qubit_to_function = {}
        self.qubit_to_inverse_function = {}

        #
        self.qubit_to_calibration_fluxes = {}
        self.qubit_to_calibration_bad_indices = {}

        self.initUI()

    def initUI(self):

        self.tab_widget = QTabWidget()
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        for qubit in self.qubits:
            tab = QWidget()

            self.tab_widget.addTab(tab, qubit)

            # layout = QVBoxLayout()

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)

            tab_content_widget = QWidget()
            tab_content_layout = QVBoxLayout(tab_content_widget)
            scroll_area.setWidget(tab_content_widget)

            tab_content_layout.addWidget(self.create_spec_fit_section(qubit))
            tab_content_layout.addWidget(self.create_calibration_section(qubit))

            tab.setLayout(QVBoxLayout())
            tab.layout().addWidget(scroll_area)

        main_layout.addWidget(self.tab_widget)

    def create_spec_fit_section(self, qubit):
        section = self.spec_fit_section
        section_widget = QGroupBox('Spec Fits')
        section_layout = QVBoxLayout()
        section_widget.setLayout(section_layout)

        # First row: File input, load button, and plot display
        file_row = QHBoxLayout()
        directory_input = QLineEdit()
        directory_input.setText(
            rf'C:\Users\mattm\OneDrive\Desktop\Research\Measurements\4Q_Triange_Lattice\Cooldown July 15, 2024\qubit_spectroscopy\{qubit}')
        load_button = QPushButton("Load")

        spec_canvas = create_plot_canvas()

        def create_load_data_lambda(_qubit, _canvas, _directory_input):
            return lambda: self.load_data(_qubit, _canvas, section, directory=_directory_input.text())

        load_button.clicked.connect(create_load_data_lambda(qubit, spec_canvas, directory_input))

        file_row.addWidget(QLabel("Filename:"))
        file_row.addWidget(directory_input)
        file_row.addWidget(load_button)

        # self.qubit_to_spec_canvas[qubit] = spec_canvas
        file_row.addWidget(spec_canvas)

        # layout.addLayout(file_row)
        section_layout.addLayout(file_row)

        # Second row: Show fits checkbox, extract frequencies button, variable plots display
        extract_frequencies_row_layout = QHBoxLayout()
        show_fits_checkbox = QCheckBox("Show fits")
        extract_frequencies_button = QPushButton("Extract Frequencies")

        # Scroll area for variable plots
        scroll_area_plots = QScrollArea()
        scroll_area_plots.setWidgetResizable(True)
        scroll_area_plots.setMinimumHeight(300)

        frequencies_canvas = create_plot_canvas()

        def create_extract_frequencies_lambda(_qubit, _canvas, _show_fits_checkbox):
            return lambda: self.extract_frequencies(_qubit, _canvas, section,
                                                    plot_fits=show_fits_checkbox.checkState() == Qt.Checked)

        extract_frequencies_button.clicked.connect(
            create_extract_frequencies_lambda(qubit, frequencies_canvas, show_fits_checkbox))

        extract_frequencies_row_layout.addWidget(show_fits_checkbox)
        extract_frequencies_row_layout.addWidget(extract_frequencies_button)

        extract_frequencies_plots_widget = QWidget()
        extract_frequencies_plots_layout = QVBoxLayout(extract_frequencies_plots_widget)
        scroll_area_plots.setWidget(extract_frequencies_plots_widget)
        extract_frequencies_plots_layout.addWidget(frequencies_canvas)

        extract_frequencies_row_layout.addWidget(scroll_area_plots)

        section_layout.addLayout(extract_frequencies_row_layout)

        # Third row: Fit frequency vs voltage and flux button, and three plots
        fit_row_layout = QHBoxLayout()
        fit_button = QPushButton("Fit Frequency vs Voltage and Flux")

        def create_fit_lambda(_qubit):
            return lambda: self.fit_frequency_vs_voltage_and_flux(_qubit)

        fit_button.clicked.connect(create_fit_lambda(qubit))

        fit_row_layout.addWidget(fit_button)

        self.qubit_to_fit_function_canvases[qubit] = [create_plot_canvas() for _ in range(3)]
        for canvas in self.qubit_to_fit_function_canvases[qubit]:
            fit_row_layout.addWidget(canvas)

        section_layout.addLayout(fit_row_layout)

        return section_widget

    def create_calibration_section(self, qubit):
        section = self.calibration_section
        section_widget = QGroupBox('Frequency to Flux conversion')
        section_layout = QVBoxLayout()
        section_widget.setLayout(section_layout)

        # First row: sweep type selection, load data button, spec plot
        file_input_row_layout = QHBoxLayout()
        filecode_input = QLineEdit()
        filecode_input.setText('0726,1523')

        # radio button group for sweep type
        sweep_type_button_group = QButtonGroup()
        sweep_type_button_group_layout = QVBoxLayout()

        sweep_type_voltage_sweep_button = QRadioButton(self.voltage_sweep)
        sweep_type_flux_sweep_button = QRadioButton(self.flux_sweep)
        sweep_type_random_voltages_button = QRadioButton(self.random_voltages)
        sweep_type_random_voltages_button.setChecked(True)
        sweep_type_button_group.addButton(sweep_type_voltage_sweep_button)
        sweep_type_button_group.addButton(sweep_type_flux_sweep_button)
        sweep_type_button_group.addButton(sweep_type_random_voltages_button)
        for button in sweep_type_button_group.buttons():
            sweep_type_button_group_layout.addWidget(button)

        load_button = QPushButton("Load")

        spec_canvas = create_plot_canvas()

        def create_load_data_lambda(_qubit, _canvas, _filecode_input, _button_group):
            return lambda: self.load_data(_qubit, _canvas, section, filecode=_filecode_input.text(),
                                          sweep_type=_button_group.checkedButton().text())

        load_button.clicked.connect(
            create_load_data_lambda(qubit, spec_canvas, filecode_input, sweep_type_button_group)
        )

        file_input_row_layout.addLayout(sweep_type_button_group_layout)
        file_input_row_layout.addWidget(filecode_input)
        file_input_row_layout.addWidget(load_button)
        file_input_row_layout.addWidget(spec_canvas)

        section_layout.addLayout(file_input_row_layout)

        ### Second row: extract frequencies button, frequencies plot

        extract_frequencies_row_layout = QHBoxLayout()
        show_fits_checkbox = QCheckBox("Show fits")
        extract_frequencies_button = QPushButton("Extract Frequencies")

        frequencies_canvas = create_plot_canvas()

        # self.qubit_to_frequencies_canvas[qubit] = frequencies_canvas

        def create_extract_frequencies_lambda(_qubit, _canvas, _show_fits_checkbox, _button_group):
            return lambda: self.extract_frequencies(_qubit, _canvas, section,
                                                    sweep_type=_button_group.checkedButton().text(),
                                                    use_ignore_indices=False,
                                                    plot_fits=show_fits_checkbox.checkState() == Qt.Checked)

        extract_frequencies_button.clicked.connect(
            create_extract_frequencies_lambda(qubit, frequencies_canvas, show_fits_checkbox, sweep_type_button_group)
        )

        extract_frequencies_row_layout.addWidget(show_fits_checkbox)
        extract_frequencies_row_layout.addWidget(extract_frequencies_button)

        # Scroll area for variable plots
        scroll_area_plots = QScrollArea()
        scroll_area_plots.setWidgetResizable(True)
        scroll_area_plots.setMinimumHeight(300)

        extract_frequencies_plots_widget = QWidget()
        extract_frequencies_plots_layout = QVBoxLayout(extract_frequencies_plots_widget)
        scroll_area_plots.setWidget(extract_frequencies_plots_widget)
        extract_frequencies_plots_layout.addWidget(frequencies_canvas)

        extract_frequencies_row_layout.addWidget(scroll_area_plots)

        # layout.addLayout(extract_frequencies_row_layout)
        section_layout.addLayout(extract_frequencies_row_layout)

        ### Third Row: convert to fluxes, button, output display

        fluxes_row_layout = QHBoxLayout()

        # convert to fluxes button
        convert_to_fluxes_button = QPushButton('Convert to fluxes')


        fluxes_row_layout.addWidget(convert_to_fluxes_button)

        # Scrollable text box
        fluxes_text_box = QTextEdit()
        fluxes_text_box.setReadOnly(True)
        text_box_scroll = QScrollArea()
        text_box_scroll.setWidget(fluxes_text_box)
        text_box_scroll.setWidgetResizable(True)
        fluxes_row_layout.addWidget(text_box_scroll)

        def create_convert_to_fluxes_lambda(_qubit, _fluxes_text_box):
            return lambda: self.convert_to_fluxes(_qubit, _fluxes_text_box)

        convert_to_fluxes_button.clicked.connect(
            create_convert_to_fluxes_lambda(qubit, fluxes_text_box)
        )

        section_layout.addLayout(fluxes_row_layout)

        return section_widget

    def load_data(self, qubit, canvas, section, filecode=None, directory=None, sweep_type=None):
        '''
        loads and plots tunable qubit spectroscopy data from matlab data file
        :param directory:
        :param qubit:
        :return:
        '''

        if sweep_type is None:
            sweep_type = self.voltage_sweep

        filepaths = []

        if directory is not None:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.mat'):
                        filepath = os.path.join(directory, file)
                        filepaths.append(filepath)
        elif filecode is not None:
            filepaths = [generate_matlab_filepath(*filecode.split(','))]
        else:
            raise ValueError('Either a directory or filecode must be provided')

        voltage_data_all = []
        frequency_data_all = []
        transmission_data_all = []

        for filepath in filepaths:
            if sweep_type == self.voltage_sweep:
                voltage_data, frequency_data, transmission_data = get_data_from_voltage_sweep_file(filepath)
            elif sweep_type in [self.flux_sweep, self.random_voltages]:
                voltage_data, frequency_data, transmission_data = get_data_from_flux_sweep_file(filepath)

            voltage_data_all.append(voltage_data)
            frequency_data_all.append(frequency_data)
            transmission_data_all.append(transmission_data)

        self.qubit_to_voltage_data[qubit][section] = voltage_data_all
        self.qubit_to_frequency_data[qubit][section] = frequency_data_all
        self.qubit_to_transmission_data[qubit][section] = transmission_data_all

        # To store axis limits
        voltage_min_all = float('inf')
        voltage_max_all = float('-inf')
        frequency_min_all = float('inf')
        frequency_max_all = float('-inf')

        # canvas = self.qubit_to_spec_canvas[qubit]
        fig = canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)

        for i in range(len(transmission_data_all)):

            if sweep_type == self.voltage_sweep:
                voltage_min = voltage_data_all[i][0]
                voltage_max = voltage_data_all[i][-1]
                voltage_step = voltage_data_all[i][1] - voltage_data_all[i][0]

                frequency_min = frequency_data_all[i][0]
                frequency_max = frequency_data_all[i][-1]
                frequency_step = frequency_data_all[i][1] - frequency_data_all[i][0]

                extent = (voltage_min - voltage_step / 2,
                          voltage_max + voltage_step / 2,
                          frequency_min - frequency_step / 2,
                          frequency_max + frequency_step / 2)

                # Update the overall min/max for voltage and frequency
                voltage_min_all = min(voltage_min_all, extent[0])
                voltage_max_all = max(voltage_max_all, extent[1])
                frequency_min_all = min(frequency_min_all, extent[2])
                frequency_max_all = max(frequency_max_all, extent[3])

            elif sweep_type in [self.flux_sweep, self.random_voltages]:
                frequency_min = frequency_data_all[i][0]
                frequency_max = frequency_data_all[i][-1]
                frequency_step = frequency_data_all[i][1] - frequency_data_all[i][0]

                extent = (0,
                          voltage_data_all[i].shape[0],
                          frequency_min - frequency_step / 2,
                          frequency_max + frequency_step / 2)

            im = ax.imshow(transmission_data_all[i], interpolation='none', vmin=-2, vmax=10, origin='lower',
                           cmap='summer', aspect='auto', extent=extent)

        if sweep_type == self.voltage_sweep:
            # Set the overall axis limits
            ax.set_xlim(voltage_min_all, voltage_max_all)
            ax.set_ylim(frequency_min_all, frequency_max_all)

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Frequency (MHz)')

        # Add colorbar
        canvas.figure.colorbar(im, ax=ax, label='Transmission (a.u.)')

        title = f'{qubit} frequency vs voltage'
        ax.set_title(title)

        canvas.draw()

    def extract_frequencies(self, qubit, canvas, section, sweep_type=None,
                            plot_fits=False, use_ignore_indices=True):
        '''

        :param qubit: qubit associated with all measurements, used as label for data dictionaries
        :param canvas: canvas to plot frequency data on
        :param output_voltages: dictionary to store extracted voltages
        :param output_frequencies: dictionary to store extracted frequencies
        :param sweep_type: 'Voltage Sweep', 'Flux Sweep', 'Random Voltages' - determines format of matlab data file
        :param plot_fits:
        :param use_ignore_indices: use manually created ignore indices for initial qubit spectroscopy to ignore bad fits
        :return:
        '''

        # # default values
        # if output_voltages is None:
        #     output_voltages = self.qubit_to_voltages
        #
        # if output_frequencies is None:
        #     output_frequencies = self.qubit_to_frequencies

        if sweep_type is None:
            sweep_type = self.voltage_sweep

        voltage_data_all = self.qubit_to_voltage_data[qubit][section]
        frequency_data_all = self.qubit_to_frequency_data[qubit][section]
        transmission_data_all = self.qubit_to_transmission_data[qubit][section]

        start_index = 0
        frequency_index_span = 50

        voltage_points_with_fit, center_frequencies, _ = extract_frequencies(voltage_data_all, frequency_data_all,
                                                                             transmission_data_all, sweep_type,
                                                                             start_index, frequency_index_span,
                                                                             plot_fits=plot_fits)

        self.qubit_to_voltages[qubit][section] = voltage_points_with_fit
        self.qubit_to_frequencies[qubit][section] = center_frequencies

        # output_voltages[qubit] = voltage_points_with_fit
        # output_frequencies[qubit] = center_frequencies

        if use_ignore_indices:
            # remove ignored indices
            ignore_indices = self.qubit_to_ignore_indices[qubit]

            # expand tuple ranges into their explicit set of indices
            ignore_indices_expanded = set()

            for i in range(len(ignore_indices)):
                ignore_index = ignore_indices[i]
                if isinstance(ignore_index, int):
                    ignore_indices_expanded.add(ignore_index)
                elif isinstance(ignore_index, (tuple, list)):
                    ignore_indices_expanded.update(range(ignore_index[0], ignore_index[1]))

            inverse_ignore_indices = set(range(len(voltage_points_with_fit))) - ignore_indices_expanded
            inverse_ignore_indices = list(inverse_ignore_indices)

            filtered_voltages = np.copy(voltage_points_with_fit)[inverse_ignore_indices]
            filtered_frequencies = np.copy(center_frequencies)[inverse_ignore_indices]

            self.qubit_to_filtered_voltages[qubit] = filtered_voltages
            self.qubit_to_filtered_frequencies[qubit] = filtered_frequencies

            voltages = filtered_voltages
            frequencies = filtered_frequencies


        else:
            voltages = voltage_points_with_fit
            frequencies = center_frequencies

        # canvas = self.qubit_to_frequencies_canvas[qubit]
        fig = canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)

        if sweep_type == self.voltage_sweep:
            ax.plot(voltage_points_with_fit, center_frequencies, marker='o', linestyle='', ms=4)
            if use_ignore_indices:
                ax.plot(filtered_voltages, filtered_frequencies, marker='o', linestyle='', ms=4)
            ax.set_xlabel('Voltage (V)')
        elif sweep_type in [self.flux_sweep, self.random_voltages]:
            ax.plot(range(voltages.shape[0]), frequencies, marker='o', linestyle='', ms=4)
            ax.set_xlabel('Voltage Index')

        ax.set_ylabel('Frequency (GHz)')

        ax.set_title(f'{qubit} Spectroscopy vs flux')

        canvas.draw()

    def fit_frequency_vs_voltage_and_flux(self, qubit):

        trianglemon_initial_guess = self.qubit_to_trianglemon_initial_guess[qubit]
        trianglemon_bounds = self.qubit_to_trianglemon_bounds[qubit]

        voltages = self.qubit_to_voltages[qubit][self.spec_fit_section]
        frequencies = self.qubit_to_frequencies[qubit][self.spec_fit_section]

        filtered_voltages = self.qubit_to_filtered_voltages[qubit]
        filtered_frequencies = self.qubit_to_filtered_frequencies[qubit]

        fit_voltages = np.linspace(voltages[0], voltages[-1], 1001)

        # if show_guess:
        #     plt.plot(fit_voltages, frequency_model_fit_trianglemon(fit_voltages, *trianglemon_initial_guess),
        #              label='guess')

        # fit
        trianglemon_popt, trianglemon_pcov = curve_fit(frequency_model_fit_trianglemon, filtered_voltages,
                                                       filtered_frequencies, p0=trianglemon_initial_guess,
                                                       bounds=trianglemon_bounds)

        # create qubit functions
        def create_qubit_function(popt):
            x0, a, b, c, d, e_1 = popt
            return lambda x: frequency_model_fit_trianglemon((np.pi * x) / b + x0, *popt)

        # create inverse qubit function
        # takes a frequency value and outputs flux
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

        self.qubit_to_function[qubit] = create_qubit_function(trianglemon_popt)
        self.qubit_to_inverse_function[qubit] = create_qubit_inverse_function(self.qubit_to_function[qubit])

        # data and fit plot
        data_and_fit_canvas = self.qubit_to_fit_function_canvases[qubit][0]

        ax = data_and_fit_canvas.figure.axes[0]
        ax.clear()

        ax.plot(filtered_voltages, filtered_frequencies, marker='o', linestyle='', ms=4, label='data')
        ax.plot(fit_voltages, frequency_model_fit_trianglemon(fit_voltages, *trianglemon_popt), label='fit')

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Frequency (GHz)')

        ax.set_title(f'{qubit} Spectroscopy vs voltage: trianglemon model')

        ax.legend()

        data_and_fit_canvas.draw()

        # frequency vs flux plot
        frequency_vs_flux_canvas = self.qubit_to_fit_function_canvases[qubit][1]

        ax = frequency_vs_flux_canvas.figure.axes[0]
        ax.clear()

        fit_fluxes = np.linspace(0, 0.5, 101)
        fit_frequencies = self.qubit_to_function[qubit](fit_fluxes)
        ax.plot(fit_fluxes, fit_frequencies)

        ax.set_xlabel('Flux')
        ax.set_ylabel('Frequency (GHz)')

        ax.set_title(f'{qubit} Spectroscopy vs flux: trianglemon model')

        frequency_vs_flux_canvas.draw()

        # flux vs frequency plot
        flux_vs_frequency_canvas = self.qubit_to_fit_function_canvases[qubit][2]

        ax = flux_vs_frequency_canvas.figure.axes[0]
        ax.clear()

        ax.plot(fit_frequencies, self.qubit_to_inverse_function[qubit](fit_frequencies))

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Flux')

        ax.set_title(f'{qubit} Flux vs frequency: trianglemon model')

        flux_vs_frequency_canvas.draw()

    def convert_to_fluxes(self, qubit, fluxes_text_box):


        qubit_inverse_function = self.qubit_to_inverse_function[qubit]

        frequency_points = self.qubit_to_frequencies[qubit]['calibration']

        bad_indices = []
        flux_points = []
        for i in range(len(frequency_points)):
            try:
                flux_point = qubit_inverse_function(frequency_points[i])
            except:
                bad_indices.append(i)
                flux_points.append(0)
            else:
                if np.isnan(flux_point):
                    flux_points.append(0)
                    bad_indices.append(i)
                else:
                    flux_points.append(flux_point)

        self.qubit_to_calibration_fluxes[qubit] = flux_points
        self.qubit_to_calibration_bad_indices[qubit] = bad_indices

        fluxes_text_box.setText(f'fluxes = {np.array_str(np.round(flux_points, 2))}'
                                f'\n\nbad indices = {bad_indices}')

class SpectroscopySubTabCoupler(QWidget):
    couplers = ['C12', 'C13', 'C23', 'C24', 'C34']

    flux_quantum_section = 'flux_quantum'
    inverse_section = 'inverse'
    calibration_section = 'calibration'

    def __init__(self, tab_name):
        super().__init__()
        self.tab_widget = None
        self.tab_name = tab_name

        ### flux quantum section parameters
        # self.coupler_to_flux_quantum_spec_canvas = {}

        self.qubit_coupler_to_voltage_data = {qubit_coupler: {} for qubit_coupler in
                                              product(SpectroscopySubTabQubit.qubits, self.couplers)}
        self.qubit_coupler_to_frequency_data = {qubit_coupler: {} for qubit_coupler in
                                                product(SpectroscopySubTabQubit.qubits, self.couplers)}
        self.qubit_coupler_to_transmission_data = {qubit_coupler: {} for qubit_coupler in
                                                   product(SpectroscopySubTabQubit.qubits, self.couplers)}

        self.qubit_coupler_to_voltages = {qubit_coupler: {} for qubit_coupler in
                                          product(SpectroscopySubTabQubit.qubits, self.couplers)}
        self.qubit_coupler_to_frequencies = {qubit_coupler: {} for qubit_coupler in
                                             product(SpectroscopySubTabQubit.qubits, self.couplers)}

        self.qubit_coupler_to_flux_quantum_voltage = {}

        ### coupler inverse section parameters
        self.coupler_to_function = {}
        self.coupler_to_inverse_function = {}

        ### default filenames
        self.default_flux_quantum_spec_directory = r'C:\Users\mattm\OneDrive\Desktop\Research\Measurements\4Q_Triange_Lattice\Cooldown July 15, 2024\coupler_spectroscopy\flux_quantum'
        self.coupler_to_default_flux_quantum_spec_filenames = {}
        self.coupler_to_default_flux_quantum_spec_filenames['C12'] = [generate_matlab_filename('0717', '1024'),
                                                                      generate_matlab_filename('0717', '1034')]
        self.coupler_to_default_flux_quantum_spec_filenames['C13'] = ['', '']
        self.coupler_to_default_flux_quantum_spec_filenames['C23'] = [generate_matlab_filename('0717', '1055'),
                                                                      generate_matlab_filename('0717', '1111')]
        self.coupler_to_default_flux_quantum_spec_filenames['C24'] = [generate_matlab_filename('0717', '1125'),
                                                                      generate_matlab_filename('0717', '1138')]
        self.coupler_to_default_flux_quantum_spec_filenames['C34'] = [generate_matlab_filename('0717', '1155'),
                                                                      generate_matlab_filename('0717', '1212')]

        self.coupler_to_default_filecode = {'C12': '0725,1705',
                                            'C13': '',
                                            'C23': '',
                                            'C24': '',
                                            'C34': '0729,0027'}

        ### calibration section parameters
        self.coupler_to_calibration_fluxes = {}
        self.coupler_to_calibration_bad_indices = {}

        self.initUI()

    def initUI(self):

        self.tab_widget = QTabWidget()
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        for coupler in self.couplers:
            tab = QWidget()

            self.tab_widget.addTab(tab, coupler)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)

            tab_content_widget = QWidget()
            tab_content_layout = QVBoxLayout(tab_content_widget)
            scroll_area.setWidget(tab_content_widget)

            # Row 0: reference qubit dropdown
            self.reference_qubit_dropdown = QComboBox()

            reference_row_layout = QHBoxLayout()

            for qubit in SpectroscopySubTabQubit.qubits:
                self.reference_qubit_dropdown.addItem(qubit)
            if coupler in ['C12', 'C23', 'C24']:
                self.reference_qubit_dropdown.setCurrentIndex(SpectroscopySubTabQubit.qubits.index('Q2'))
            elif coupler in ['C13', 'C34']:
                self.reference_qubit_dropdown.setCurrentIndex(SpectroscopySubTabQubit.qubits.index('Q3'))

            reference_row_layout.addWidget(QLabel('reference qubit:'))
            reference_row_layout.addWidget(self.reference_qubit_dropdown)
            tab_content_layout.addLayout(reference_row_layout)

            tab_content_layout.addWidget(self.create_flux_quantum_fit_section(coupler))
            tab_content_layout.addWidget(self.create_coupler_inverse_fit_section(coupler))
            tab_content_layout.addWidget(self.create_calibration_section(coupler))

            tab.setLayout(QVBoxLayout())
            tab.layout().addWidget(scroll_area)

        main_layout.addWidget(self.tab_widget)

    def create_flux_quantum_fit_section(self, coupler):
        section = self.flux_quantum_section
        section_widget = QGroupBox('Flux Quantum Fits')
        section_layout = QVBoxLayout()
        section_widget.setLayout(section_layout)

        # First row: 2 File inputs, load button, and plot display

        file_row = QHBoxLayout()
        filename_input_1 = QLineEdit()
        filename_input_2 = QLineEdit()

        filename_input_1.setText(os.path.join(self.default_flux_quantum_spec_directory, coupler,
                                              self.coupler_to_default_flux_quantum_spec_filenames[coupler][0]))
        filename_input_2.setText(os.path.join(self.default_flux_quantum_spec_directory, coupler,
                                              self.coupler_to_default_flux_quantum_spec_filenames[coupler][1]))

        load_button = QPushButton("Load")

        spec_canvas = create_plot_canvas()

        def create_load_data_lambda(_filename_input_1, _filename_input_2, _reference_qubit_dropdown, _coupler, _canvas):
            return lambda: self.load_data(_reference_qubit_dropdown.currentText(), _coupler, _canvas, section,
                                          filenames=[_filename_input_1.text(), _filename_input_2.text()])

        load_button.clicked.connect(
            create_load_data_lambda(filename_input_1, filename_input_2, self.reference_qubit_dropdown,
                                    coupler, spec_canvas)
        )

        file_row.addWidget(QLabel("Filename:"))
        file_row.addWidget(filename_input_1)
        file_row.addWidget(filename_input_2)
        file_row.addWidget(load_button)

        # self.coupler_to_flux_quantum_spec_canvas[coupler] = spec_canvas
        file_row.addWidget(spec_canvas)

        # layout.addLayout(file_row)
        section_layout.addLayout(file_row)

        # Second row: Show fits checkbox, extract frequencies button, fit data button, variable plots display
        extract_frequencies_row_layout = QHBoxLayout()
        show_fits_checkbox = QCheckBox("Show fits")

        buttons_layout = QVBoxLayout()
        extract_frequencies_button = QPushButton("Extract Frequencies")

        frequencies_canvas = create_plot_canvas()

        def create_extract_frequencies_lambda(_reference_qubit_dropdown, _coupler, _show_fits_checkbox, _canvas):
            return lambda: self.extract_frequencies(_reference_qubit_dropdown.currentText(), _coupler, _canvas, section,
                                                    plot_fits=show_fits_checkbox.checkState() == Qt.Checked)

        extract_frequencies_button.clicked.connect(
            create_extract_frequencies_lambda(self.reference_qubit_dropdown, coupler, show_fits_checkbox,
                                              frequencies_canvas)
        )

        extract_frequencies_row_layout.addWidget(show_fits_checkbox)

        # Scroll area for variable plots
        scroll_area_plots = QScrollArea()
        scroll_area_plots.setWidgetResizable(True)
        scroll_area_plots.setMinimumHeight(300)

        extract_frequencies_plots_widget = QWidget()
        extract_frequencies_plots_layout = QVBoxLayout(extract_frequencies_plots_widget)
        # extract_frequencies_plots_widget.setLayout(extract_frequencies_plots_layout)
        scroll_area_plots.setWidget(extract_frequencies_plots_widget)

        ### fit button
        fit_data_button = QPushButton("Fit Data")
        flux_quantum_label = QLabel('Flux Quantum:      ')

        def create_fit_data_button_lambda(_reference_qubit_dropdown, _coupler, _flux_quantum_label, _canvas):
            return lambda: self.fit_two_coupler_peaks(_reference_qubit_dropdown.currentText(), _coupler,
                                                      _flux_quantum_label, _canvas)

        fit_data_button.clicked.connect(
            create_fit_data_button_lambda(self.reference_qubit_dropdown, coupler, flux_quantum_label,
                                          frequencies_canvas)
        )

        buttons_layout.addWidget(extract_frequencies_button)
        buttons_layout.addWidget(fit_data_button)
        buttons_layout.addWidget(flux_quantum_label)
        extract_frequencies_row_layout.addLayout(buttons_layout)

        extract_frequencies_row_layout.addWidget(scroll_area_plots)

        # self.coupler_to_flux_quantum_frequencies_canvas[coupler] = frequencies_canvas
        extract_frequencies_plots_layout.addWidget(frequencies_canvas)

        # layout.addLayout(extract_frequencies_row_layout)
        section_layout.addLayout(extract_frequencies_row_layout)

        return section_widget

    def create_coupler_inverse_fit_section(self, coupler):
        section = self.inverse_section
        section_widget = QGroupBox('Coupler Inverse Fit')
        section_layout = QVBoxLayout()
        section_widget.setLayout(section_layout)

        # First row: 2 File inputs, load button, and plot display

        file_row = QHBoxLayout()
        filename_input = QLineEdit()

        filename_input.setText(self.coupler_to_default_filecode[coupler])

        # datecode, timecode = self.coupler_to_default_filecode[coupler].split(',')
        # filename_input.setText(generate_matlab_filepath(datecode, timecode))

        load_button = QPushButton("Load")

        spec_canvas = create_plot_canvas()

        def create_load_data_lambda(_filename_input, _reference_qubit_dropdown, _coupler, _canvas):
            return lambda: self.load_data(_reference_qubit_dropdown.currentText(), _coupler, _canvas, section,
                                          filecodes=[_filename_input.text()])

        load_button.clicked.connect(
            create_load_data_lambda(filename_input, self.reference_qubit_dropdown,
                                    coupler, spec_canvas)
        )

        file_row.addWidget(QLabel("Filename:"))
        file_row.addWidget(filename_input)
        file_row.addWidget(load_button)

        # self.coupler_to_flux_quantum_spec_canvas[coupler] = spec_canvas
        file_row.addWidget(spec_canvas)

        # layout.addLayout(file_row)
        section_layout.addLayout(file_row)

        # Second row: Show fits checkbox, extract frequencies button, fit data button, variable plots display
        extract_frequencies_row_layout = QHBoxLayout()
        show_fits_checkbox = QCheckBox("Show fits")

        buttons_layout = QVBoxLayout()
        extract_frequencies_button = QPushButton("Extract Frequencies")

        frequencies_canvas = create_plot_canvas()

        def create_extract_frequencies_lambda(_reference_qubit_dropdown, _coupler, _show_fits_checkbox, _canvas):
            return lambda: self.extract_frequencies(_reference_qubit_dropdown.currentText(), _coupler, _canvas, section,
                                                    plot_fits=show_fits_checkbox.checkState() == Qt.Checked)

        extract_frequencies_button.clicked.connect(
            create_extract_frequencies_lambda(self.reference_qubit_dropdown, coupler, show_fits_checkbox,
                                              frequencies_canvas)
        )

        extract_frequencies_row_layout.addWidget(show_fits_checkbox)

        # Scroll area for variable plots
        scroll_area_plots = QScrollArea()
        scroll_area_plots.setWidgetResizable(True)
        scroll_area_plots.setMinimumHeight(300)

        extract_frequencies_plots_widget = QWidget()
        extract_frequencies_plots_layout = QVBoxLayout(extract_frequencies_plots_widget)
        # extract_frequencies_plots_widget.setLayout(extract_frequencies_plots_layout)
        scroll_area_plots.setWidget(extract_frequencies_plots_widget)

        ### fit button
        fit_data_button = QPushButton("Fit Data")

        buttons_layout.addWidget(extract_frequencies_button)
        buttons_layout.addWidget(fit_data_button)
        extract_frequencies_row_layout.addLayout(buttons_layout)

        extract_frequencies_row_layout.addWidget(scroll_area_plots)

        # self.coupler_to_flux_quantum_frequencies_canvas[coupler] = frequencies_canvas
        extract_frequencies_plots_layout.addWidget(frequencies_canvas)

        # layout.addLayout(extract_frequencies_row_layout)
        section_layout.addLayout(extract_frequencies_row_layout)

        # Third row: coupler forward function, inverse function
        fit_row_layout = QHBoxLayout()

        coupler_function_canvas = [create_plot_canvas() for _ in range(3)]
        for canvas in coupler_function_canvas:
            fit_row_layout.addWidget(canvas)

        def create_fit_data_button_lambda(_reference_qubit_dropdown, _coupler, _canvases):
            return lambda: self.fit_coupler_functions(_reference_qubit_dropdown.currentText(), _coupler, _canvases)

        fit_data_button.clicked.connect(
            create_fit_data_button_lambda(self.reference_qubit_dropdown, coupler,
                                          [frequencies_canvas] + coupler_function_canvas)
        )

        section_layout.addLayout(fit_row_layout)

        return section_widget

    def create_calibration_section(self, coupler):
        section = self.calibration_section
        section_widget = QGroupBox('Frequency to Flux conversion')
        section_layout = QVBoxLayout()
        section_widget.setLayout(section_layout)

        # First row: sweep type selection, load data button, spec plot
        file_input_row_layout = QHBoxLayout()
        filecode_input = QLineEdit()
        filecode_input.setText('0726,1709')
        filecode_input.setText('0726,1523')

        # radio button group for sweep type
        sweep_type_button_group = QButtonGroup()
        sweep_type_button_group_layout = QVBoxLayout()

        sweep_type_voltage_sweep_button = QRadioButton(SpectroscopySubTabQubit.voltage_sweep)
        sweep_type_flux_sweep_button = QRadioButton(SpectroscopySubTabQubit.flux_sweep)
        sweep_type_random_voltages_button = QRadioButton(SpectroscopySubTabQubit.random_voltages)
        sweep_type_random_voltages_button.setChecked(True)
        sweep_type_button_group.addButton(sweep_type_voltage_sweep_button)
        sweep_type_button_group.addButton(sweep_type_flux_sweep_button)
        sweep_type_button_group.addButton(sweep_type_random_voltages_button)
        for button in sweep_type_button_group.buttons():
            sweep_type_button_group_layout.addWidget(button)

        load_button = QPushButton("Load")

        spec_canvas = create_plot_canvas()

        def create_load_data_lambda(_reference_qubit_dropdown, _coupler, _canvas, _filecode_input, _button_group):
            return lambda: self.load_data(_reference_qubit_dropdown.currentText(), _coupler, _canvas, section,
                                          filecodes=[_filecode_input.text()],
                                          sweep_type=_button_group.checkedButton().text())

        load_button.clicked.connect(
            create_load_data_lambda(self.reference_qubit_dropdown, coupler, spec_canvas, filecode_input,
                                    sweep_type_button_group)
        )

        file_input_row_layout.addLayout(sweep_type_button_group_layout)
        file_input_row_layout.addWidget(filecode_input)
        file_input_row_layout.addWidget(load_button)
        file_input_row_layout.addWidget(spec_canvas)

        section_layout.addLayout(file_input_row_layout)

        ### Second row: extract frequencies button, frequencies plot

        extract_frequencies_row_layout = QHBoxLayout()
        show_fits_checkbox = QCheckBox("Show fits")
        extract_frequencies_button = QPushButton("Extract Frequencies")

        frequencies_canvas = create_plot_canvas()

        # self.qubit_to_frequencies_canvas[qubit] = frequencies_canvas

        def create_extract_frequencies_lambda(_reference_qubit_dropdown, _coupler, _canvas, _show_fits_checkbox,
                                              _button_group):
            return lambda: self.extract_frequencies(_reference_qubit_dropdown.currentText(), _coupler, _canvas, section,
                                                    sweep_type=_button_group.checkedButton().text(),
                                                    plot_fits=show_fits_checkbox.checkState() == Qt.Checked)

        extract_frequencies_button.clicked.connect(
            create_extract_frequencies_lambda(self.reference_qubit_dropdown, coupler, frequencies_canvas,
                                              show_fits_checkbox, sweep_type_button_group)
        )

        extract_frequencies_row_layout.addWidget(show_fits_checkbox)
        extract_frequencies_row_layout.addWidget(extract_frequencies_button)

        # Scroll area for variable plots
        scroll_area_plots = QScrollArea()
        scroll_area_plots.setWidgetResizable(True)
        scroll_area_plots.setMinimumHeight(300)

        extract_frequencies_plots_widget = QWidget()
        extract_frequencies_plots_layout = QVBoxLayout(extract_frequencies_plots_widget)
        scroll_area_plots.setWidget(extract_frequencies_plots_widget)
        extract_frequencies_plots_layout.addWidget(frequencies_canvas)

        extract_frequencies_row_layout.addWidget(scroll_area_plots)

        # layout.addLayout(extract_frequencies_row_layout)
        section_layout.addLayout(extract_frequencies_row_layout)

        ### Third Row: convert to fluxes, button, output display

        fluxes_row_layout = QHBoxLayout()

        # inverse type (positive/negative) buttons
        inverse_type_button_group = QButtonGroup()
        inverse_type_button_group_layout = QVBoxLayout()

        positive_inverse_button = QRadioButton('Positive inverse')
        negative_inverse_button = QRadioButton('Negative inverse')
        positive_inverse_button.setChecked(True)
        inverse_type_button_group.addButton(positive_inverse_button)
        inverse_type_button_group.addButton(negative_inverse_button)
        for button in inverse_type_button_group.buttons():
            inverse_type_button_group_layout.addWidget(button)

        fluxes_row_layout.addLayout(inverse_type_button_group_layout)

        # convert to fluxes button
        convert_to_fluxes_button = QPushButton('Convert to fluxes')

        fluxes_row_layout.addWidget(convert_to_fluxes_button)

        # Scrollable text box
        fluxes_text_box = QTextEdit()
        fluxes_text_box.setReadOnly(True)
        text_box_scroll = QScrollArea()
        text_box_scroll.setWidget(fluxes_text_box)
        text_box_scroll.setWidgetResizable(True)
        fluxes_row_layout.addWidget(text_box_scroll)

        def create_convert_to_fluxes_lambda(_reference_qubit_dropdown, _coupler, _fluxes_text_box, _button_group):
            return lambda: self.convert_to_fluxes(_reference_qubit_dropdown.currentText(), _coupler, _fluxes_text_box,
                                                  _button_group.checkedButton().text())

        convert_to_fluxes_button.clicked.connect(
            create_convert_to_fluxes_lambda(self.reference_qubit_dropdown, coupler, fluxes_text_box,
                                            inverse_type_button_group)
        )

        section_layout.addLayout(fluxes_row_layout)

        return section_widget

    def load_data(self, qubit, coupler, canvas, section, filecodes=None, filenames=None, sweep_type=None):
        '''
        loads and plots qubit spectroscopy vs coupler flux data from matlab data file
        :param qubit:
        :param canvas:
        :param filenames:
        :param coupler:
        :param section: either 'flux_quantum', 'inverse', or 'calibration'
        :return:
        '''

        if sweep_type is None:
            sweep_type = SpectroscopySubTabQubit.voltage_sweep

        qubit_coupler = qubit, coupler

        voltage_data_all = []
        frequency_data_all = []
        transmission_data_all = []

        if filecodes is not None:
            filenames = []
            for filecode in filecodes:
                filenames.append(generate_matlab_filepath(*filecode.split(',')))

        for filename in filenames:

            if sweep_type == SpectroscopySubTabQubit.voltage_sweep:
                voltage_data, frequency_data, transmission_data = get_data_from_voltage_sweep_file(filename)
            elif sweep_type in [SpectroscopySubTabQubit.flux_sweep, SpectroscopySubTabQubit.random_voltages]:
                voltage_data, frequency_data, transmission_data = get_data_from_flux_sweep_file(filename)

            voltage_data_all.append(voltage_data)
            frequency_data_all.append(frequency_data)
            transmission_data_all.append(transmission_data)

        self.qubit_coupler_to_voltage_data[qubit_coupler][section] = voltage_data_all
        self.qubit_coupler_to_frequency_data[qubit_coupler][section] = frequency_data_all
        self.qubit_coupler_to_transmission_data[qubit_coupler][section] = transmission_data_all

        # To store axis limits
        voltage_min_all = float('inf')
        voltage_max_all = float('-inf')
        frequency_min_all = float('inf')
        frequency_max_all = float('-inf')

        # canvas = self.coupler_to_flux_quantum_spec_canvas[coupler]
        fig = canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)

        for i in range(len(transmission_data_all)):
            if sweep_type == SpectroscopySubTabQubit.voltage_sweep:
                voltage_min = voltage_data_all[i][0]
                voltage_max = voltage_data_all[i][-1]
                voltage_step = voltage_data_all[i][1] - voltage_data_all[i][0]

                frequency_min = frequency_data_all[i][0]
                frequency_max = frequency_data_all[i][-1]
                frequency_step = frequency_data_all[i][1] - frequency_data_all[i][0]

                extent = (voltage_min - voltage_step / 2,
                          voltage_max + voltage_step / 2,
                          frequency_min - frequency_step / 2,
                          frequency_max + frequency_step / 2)

                # Update the overall min/max for voltage and frequency
                voltage_min_all = min(voltage_min_all, extent[0])
                voltage_max_all = max(voltage_max_all, extent[1])
                frequency_min_all = min(frequency_min_all, extent[2])
                frequency_max_all = max(frequency_max_all, extent[3])

            elif sweep_type in [SpectroscopySubTabQubit.flux_sweep, SpectroscopySubTabQubit.random_voltages]:
                frequency_min = frequency_data_all[i][0]
                frequency_max = frequency_data_all[i][-1]
                frequency_step = frequency_data_all[i][1] - frequency_data_all[i][0]

                extent = (0,
                          voltage_data_all[i].shape[0],
                          frequency_min - frequency_step / 2,
                          frequency_max + frequency_step / 2)

            im = ax.imshow(transmission_data_all[i], interpolation='none', vmin=-2, vmax=10, origin='lower',
                           cmap='summer', aspect='auto', extent=extent)

        # Set the overall axis limits
        if sweep_type == SpectroscopySubTabQubit.voltage_sweep:
            ax.set_xlim(voltage_min_all, voltage_max_all)
            ax.set_ylim(frequency_min_all, frequency_max_all)

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Frequency (MHz)')

        # Add colorbar
        canvas.figure.colorbar(im, ax=ax, label='Transmission (a.u.)')

        title = f'{qubit} frequency vs {coupler} voltage'
        ax.set_title(title)

        canvas.draw()

    def extract_frequencies(self, qubit, coupler, canvas, section, sweep_type=None, plot_fits=False):

        if sweep_type is None:
            sweep_type = SpectroscopySubTabQubit.voltage_sweep

        qubit_coupler = qubit, coupler

        voltage_data_all = self.qubit_coupler_to_voltage_data[qubit_coupler][section]
        frequency_data_all = self.qubit_coupler_to_frequency_data[qubit_coupler][section]
        transmission_data_all = self.qubit_coupler_to_transmission_data[qubit_coupler][section]

        start_index = 0
        frequency_index_span = 50

        # canvas = self.coupler_to_frequencies_canvas[coupler]
        fig = canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)

        if section == 'flux_quantum':
            self.qubit_coupler_to_voltages[qubit_coupler][section] = []
            self.qubit_coupler_to_frequencies[qubit_coupler][section] = []
            for i in range(len(voltage_data_all)):
                voltage_points_with_fit, center_frequencies, _ = extract_frequencies([voltage_data_all[i]],
                                                                                     [frequency_data_all[i]],
                                                                                     [transmission_data_all[i]],
                                                                                     sweep_type=sweep_type,
                                                                                     start_index=start_index,
                                                                                     frequency_index_span=frequency_index_span,
                                                                                     plot_fits=plot_fits)
                self.qubit_coupler_to_voltages[qubit_coupler][section].append(voltage_points_with_fit)
                self.qubit_coupler_to_frequencies[qubit_coupler][section].append(center_frequencies)
                ax.plot(voltage_points_with_fit, center_frequencies, marker='o', linestyle='', ms=4)
        elif section in ['inverse', 'calibration']:
            voltage_points_with_fit, center_frequencies, _ = extract_frequencies(voltage_data_all,
                                                                                 frequency_data_all,
                                                                                 transmission_data_all,
                                                                                 sweep_type=sweep_type,
                                                                                 start_index=start_index,
                                                                                 frequency_index_span=frequency_index_span,
                                                                                 plot_fits=plot_fits)
            self.qubit_coupler_to_voltages[qubit_coupler][section] = voltage_points_with_fit
            self.qubit_coupler_to_frequencies[qubit_coupler][section] = center_frequencies

            if sweep_type == SpectroscopySubTabQubit.voltage_sweep:
                ax.plot(voltage_points_with_fit, center_frequencies, marker='o', linestyle='', ms=4)
                ax.set_xlabel('Voltage (V)')
            elif sweep_type in [SpectroscopySubTabQubit.flux_sweep, SpectroscopySubTabQubit.random_voltages]:
                ax.plot(range(voltage_points_with_fit.shape[0]), center_frequencies, marker='o', linestyle='', ms=4)
                ax.set_xlabel('Voltage Index')

        ax.set_ylabel('Frequency (GHz)')

        ax.set_title(f'{qubit} spectroscopy vs {coupler} flux')

        canvas.draw()

    def fit_two_coupler_peaks(self, qubit, coupler, flux_quantum_label, canvas):
        '''
        Fit two coupler peaks to lorentzians and determine their spacing to extract this coupler's flux quantum
        :param flux_quantum_label:
        :param qubit:
        :param coupler:
        :return:
        '''

        qubit_coupler = qubit, coupler

        qubit_coupler_voltages = self.qubit_coupler_to_voltages[qubit_coupler][self.flux_quantum_section]
        qubit_coupler_frequencies = self.qubit_coupler_to_frequencies[qubit_coupler][self.flux_quantum_section]

        peak_voltages = []

        fig = canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)

        for i in range(len(qubit_coupler_voltages)):
            popt = fit_coupler_peak_to_lorentzian(qubit_coupler_voltages[i], qubit_coupler_frequencies[i])
            peak_voltages.append(popt[0])

            fit_voltages = np.linspace(qubit_coupler_voltages[i][0], qubit_coupler_voltages[i][-1], 101)
            fit_frequencies = lorentzian_fit(fit_voltages, *popt)

            # plot fit

            voltages = qubit_coupler_voltages[i]
            frequencies = qubit_coupler_frequencies[i]

            ax.plot(voltages, frequencies, marker='o', linestyle='', ms=4, label=f'Peak {i + 1} data')
            ax.plot(fit_voltages, fit_frequencies, label=f'Peak {i + 1} fit')

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Frequency (GHz)')

        ax.set_title(f'{qubit} spectroscopy vs {coupler} flux')
        ax.legend()

        canvas.draw()

        flux_quantum_voltage = peak_voltages[1] - peak_voltages[0]
        self.qubit_coupler_to_flux_quantum_voltage[qubit_coupler] = abs(flux_quantum_voltage)

        flux_quantum_label.setText(f'Flux Quantum: {np.round(flux_quantum_voltage, 3)} V')

    def fit_coupler_functions(self, qubit, coupler, canvases):

        coupler_voltages = self.qubit_coupler_to_voltages[(qubit, coupler)][self.inverse_section]
        coupler_frequencies = self.qubit_coupler_to_frequencies[(qubit, coupler)][self.inverse_section]

        popt = fit_coupler_peak_to_lorentzian(coupler_voltages, coupler_frequencies)

        ### fit
        fig = canvases[0].figure
        fig.clf()
        ax = fig.add_subplot(111)

        fit_voltages = np.linspace(coupler_voltages[0], coupler_voltages[-1], 101)
        fit_frequencies = lorentzian_fit(fit_voltages, *popt)

        ax.plot(coupler_voltages, coupler_frequencies, marker='o', linestyle='', ms=4, label=f'data')
        ax.plot(fit_voltages, fit_frequencies, label=f'fit')

        ax.set_title('Forward function')

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Frequency (GHz)')
        ax.legend()
        canvases[0].draw()

        def create_coupler_function(_popt, _flux_quantum):
            # in terms of flux
            return lambda x: lorentzian_fit(x * _flux_quantum + popt[0], *_popt)

        def create_coupler_inverse_function(_popt, _flux_quantum, positive=True):
            return lambda x: (lorentzian_inverse(x, *_popt, positive=positive) - popt[0]) / _flux_quantum

        ### forward function
        flux_quantum = self.qubit_coupler_to_flux_quantum_voltage[(qubit, coupler)]
        self.coupler_to_function[coupler] = create_coupler_function(popt, flux_quantum)

        # coupler_to_peak_voltage[coupler] = popt[0]

        fig = canvases[1].figure
        fig.clf()
        ax = fig.add_subplot(111)

        fit_fluxes = np.linspace(-0.1, 0.1, 101)
        fit_frequencies = self.coupler_to_function[coupler](fit_fluxes)
        ax.plot(fit_fluxes, fit_frequencies)

        ax.set_title('Forward function')

        ax.set_xlabel('Flux')
        ax.set_ylabel('Frequency (GHz)')
        ax.legend()
        canvases[1].draw()

        # inverse function
        # two functions depending on sign of square root

        self.coupler_to_inverse_function[coupler] = {}

        self.coupler_to_inverse_function[coupler]['positive'] = create_coupler_inverse_function(popt, flux_quantum,
                                                                                                True)
        self.coupler_to_inverse_function[coupler]['negative'] = create_coupler_inverse_function(popt, flux_quantum,
                                                                                                False)

        fig = canvases[2].figure
        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(fit_frequencies, self.coupler_to_inverse_function[coupler]['positive'](fit_frequencies),
                label='positive')
        ax.plot(fit_frequencies, self.coupler_to_inverse_function[coupler]['negative'](fit_frequencies),
                label='negative')

        ax.set_title('Inverse function')
        ax.legend()

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Flux')
        canvases[2].draw()

        fig = canvases[3].figure
        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(fit_voltages, self.coupler_to_inverse_function[coupler]['positive'](fit_frequencies), label='positive')
        ax.plot(fit_voltages, self.coupler_to_inverse_function[coupler]['negative'](fit_frequencies), label='negative')

        ax.set_title('Verify inverse')
        ax.legend()

        ax.set_xlabel('Flux')
        ax.set_ylabel('Flux')
        canvases[3].draw()

    def convert_to_fluxes(self, qubit, coupler, fluxes_text_box, inverse_type='positive'):

        if 'positive' in inverse_type.lower():
            inverse_type = 'positive'
        elif 'negative' in inverse_type.lower():
            inverse_type = 'negative'

        coupler_inverse_function = self.coupler_to_inverse_function[coupler][inverse_type]

        frequency_points = self.qubit_coupler_to_frequencies[(qubit, coupler)]['calibration']
        bad_indices = []
        flux_points = []
        for i in range(len(frequency_points)):
            try:
                flux_point = coupler_inverse_function(frequency_points[i])
            except:
                bad_indices.append(i)
                flux_points.append(0)
            else:
                if np.isnan(flux_point):
                    flux_points.append(0)
                    bad_indices.append(i)
                else:
                    flux_points.append(flux_point)

        self.coupler_to_calibration_fluxes[coupler] = flux_points
        self.coupler_to_calibration_bad_indices[coupler] = bad_indices

        fluxes_text_box.setText(f'fluxes = {np.array_str(np.round(flux_points, 2))}'
                                f'\n\nbad indices = {bad_indices}')

def lorentzian_fit(x, x0, a, b, c):
    return a / (b + np.power((x - x0), 2)) + c


def lorentzian_inverse(x, x0, a, b, c, positive=True):
    if positive:
        return x0 + np.sqrt(a / (x - c) - b)
    else:
        return x0 - np.sqrt(a / (x - c) - b)


def frequency_model_fit_trianglemon(x, x0, a, b, c, d, e):
    EJ = np.sqrt(np.power(np.cos(b * (x - x0)), 2) + (d ** 2) * np.power(np.sin(b * (x - x0)), 2))

    return np.sqrt(a * EJ / (1 + e * EJ)) - c / np.power(1 + e * EJ, 3)


def get_data_from_voltage_sweep_file(filepath):
    data1 = loadmat(filepath)
    specAmpData1 = data1['specamp']
    specPhaseData1 = data1['specphase']
    specFreqVector1 = data1['specfreq']
    volts1 = data1['voltage_vector']

    volts1 = np.asarray(volts1[0])
    specFreqVector1 = np.asarray(specFreqVector1[0]) * 1e-9

    # Xlist = (Xlist - voltOffSet +VoltPerFlux*center)/VoltPerFlux

    # X, Y = np.meshgrid(Xlist, Ylist)

    ### make copies of spec data
    phase = specPhaseData1.copy()
    amp = specAmpData1.copy()

    ### remove average for better plotting
    for i in range(0, len(phase[:, 1])):
        phase[i, :] = phase[i, :] - np.mean(phase[i, :])
        amp[i, :] = amp[i, :] - np.mean(amp[i, :])
    amp = amp  # [::-1]
    Z = amp.copy()
    Z = np.asarray(Z)
    Z = np.transpose(Z)

    return volts1, specFreqVector1, Z


def get_data_from_flux_sweep_file(filepath):
    '''
    Since voltages are different at each point for each channel, each point in the sweep isare labeled by an index
    rather than a voltage value
    :param filepath:
    :return:
    '''
    data1 = loadmat(filepath)
    transAmpData1 = data1['transamp']
    specAmpData1 = data1['specamp']
    specPhaseData1 = data1['specphase']
    specFreqVector1 = data1['specfreq']
    random_voltages = data1['voltage_matrix']

    random_voltages = np.asarray(random_voltages)
    specFreqVector1 = np.asarray(specFreqVector1[0]) * 1e-9

    ### make copies of spec data
    phase = specPhaseData1.copy()
    amp = specAmpData1.copy()

    ### remove average for better plotting
    for i in range(0, len(phase[:, 1])):
        phase[i, :] = phase[i, :] - np.mean(phase[i, :])
        amp[i, :] = amp[i, :] - np.mean(amp[i, :])
    amp = amp  # [::-1]
    Z = amp.copy()
    Z = np.asarray(Z)
    Z = np.transpose(Z)

    return random_voltages, specFreqVector1, Z


def create_plot_canvas(min_size=(300, 250)):
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    canvas.setMinimumSize(*min_size)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return canvas


def extract_frequencies(voltage_data_all, frequency_data_all, transmission_data_all, sweep_type, start_index=5,
                        frequency_index_span=50, plot_fits=False):
    voltage_points_with_fit = []
    center_frequencies = []
    center_frequency_errors = []

    for j in range(len(voltage_data_all)):

        voltages = voltage_data_all[j]

        for i in range(len(voltages)):

            frequencies = frequency_data_all[j]

            row = transmission_data_all[j][start_index:, i]

            peak_index = np.argmax(row) + start_index
            center_frequency_guess = frequencies[peak_index]

            # fit to lorentzian
            # restrict fit in range span around peak

            restricted_frequencies = frequencies[max(peak_index - frequency_index_span // 2, 0):min(
                peak_index + frequency_index_span // 2, len(frequencies))]
            restricted_row = transmission_data_all[j][
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
                # if it fails, plot the data it was trying to fit
                # plt.plot(restricted_frequencies, filtered_row, linestyle='', marker='o', label='data')
                # plt.plot(restricted_frequencies, lorentzian_fit(restricted_frequencies, *initial_guess), label='guess')
                # plt.xlabel('Frequency (MHz)')
                # plt.title(f'Lorentzian fit for index {i}')
                # plt.axvline(center_frequency_guess, color='red')
                # plt.legend()
                # plt.show()

                print('Couldn\'t get a fit')

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
                    # plt.plot(frequencies[start_index:], row, linestyle='', marker='o', label='data')
                    #
                    # fit_frequencies = np.linspace(frequencies[start_index], frequencies[-1], 1000)
                    # plt.plot(fit_frequencies, lorentzian_fit(fit_frequencies, *popt), label='fit')
                    # plt.axvline(center_frequency_guess, color='red')
                    # plt.legend()
                    #
                    # plt.xlabel('Frequency (MHz)')
                    # plt.title(f'Lorentzian fit for index {i}')
                    # plt.show()

                    print(f'Center frequency is {popt[0]} MHz')

    # sort voltages
    voltage_points_with_fit = np.array(voltage_points_with_fit)
    center_frequencies = np.array(center_frequencies)
    center_frequency_errors = np.array(center_frequency_errors)

    if sweep_type == SpectroscopySubTabQubit.voltage_sweep:
        sorted_indices = voltage_points_with_fit.argsort()
        voltage_points_with_fit = voltage_points_with_fit[sorted_indices]
        center_frequencies = center_frequencies[sorted_indices]
        center_frequency_errors = center_frequency_errors[sorted_indices]

    return voltage_points_with_fit, center_frequencies, center_frequency_errors


def fit_coupler_peak_to_lorentzian(voltages, frequencies, plot_fits=False):
    peak_index_guess = np.argmax(-frequencies)
    peak_voltage_guess = voltages[peak_index_guess]

    bounds = ([voltages[0], -np.inf, 0, -np.inf], [voltages[-1], np.inf, np.inf, np.inf])
    initial_guess = [peak_voltage_guess, -1e-5, 0.0001, 5.25]

    popt, pcov = curve_fit(lorentzian_fit, voltages, frequencies, p0=initial_guess, bounds=bounds)

    if plot_fits:
        plt.plot(voltages, frequencies, linestyle='', marker='o', label='data')
        #     plt.plot(voltages, lorentzian_fit(voltages, *initial_guess), label='guess')

        fit_voltages = np.linspace(voltages[0], voltages[-1], 1001)
        plt.plot(fit_voltages, lorentzian_fit(fit_voltages, *popt), label='fit')
        plt.xlabel('Frequency (MHz)')
        plt.title(f'Lorentzian fit')
        plt.axvline(peak_voltage_guess, color='red', linestyle=':')
        plt.legend()
        plt.show()

    return popt


def generate_matlab_filename(datecode, timecode):
    return r'2Tone4Qubit_NR_2024{}_{}'.format(datecode, timecode)


def generate_matlab_filepath(datecode, timecode):
    return r'V:\QSimMeasurements\Measurements\4Q_Triangle_Lattice\pnax{}24\2Tone4Qubit_NR_2024{}_{}'.format(datecode,
                                                                                                            datecode,
                                                                                                            timecode)
