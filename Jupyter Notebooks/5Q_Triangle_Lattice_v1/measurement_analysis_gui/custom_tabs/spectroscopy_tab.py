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

from custom_widgets.load_files_widget import LoadFilesWidget
from custom_widgets.manual_filter_data_widget import ManualFilterDataWidget
from custom_widgets.peak_fit_widget import PeakFitWidget
from custom_widgets.fit_transmon import FitTransmonWidget

class SpectroscopyMeasurementsTab(QWidget):
    def __init__(self, root_directory, measurements_directory):
        super().__init__()

        self.root_directory = root_directory
        self.measurements_directory = measurements_directory

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        tabs = QTabWidget()

        self.qubits_tab = SpectroscopySubTabQubit("Qubit Spectroscopy", self.root_directory, self.measurements_directory)
        # self.couplers_tab = SpectroscopySubTabCoupler("Coupler Spectroscopy")

        tabs.addTab(self.qubits_tab, "Qubit Spectroscopy")
        # tabs.addTab(self.couplers_tab, "Coupler Spectroscopy")

        layout.addWidget(tabs)
        self.setLayout(layout)


class SpectroscopySubTabQubit(QWidget):
    qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

    def __init__(self, tab_name, root_directory, measurements_directory):
        super().__init__()
        self.tab_name = tab_name
        self.root_directory = root_directory

        self.qubit_spectroscopy_directory = os.path.join(self.root_directory, 'data\qubit_spectroscopy')
        # self.default_config_directory = os.path.join(self.root_directory, 'data\qubit_spectroscopy')

        self.measurements_directory = measurements_directory

        self.initUI()

    def initUI(self):

        self.tab_widget = QTabWidget()
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        for qubit in self.qubits:

            print(f'{qubit}')

            tab = QWidget()

            self.tab_widget.addTab(tab, qubit)

            # layout = QVBoxLayout()

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)

            tab_content_widget = QWidget()
            tab_content_layout = QVBoxLayout(tab_content_widget)
            scroll_area.setWidget(tab_content_widget)

            ### default filepaths
            # script_directory = os.path.dirname(os.path.abspath(__file__))
            # qubit_default_files_config_filename = os.path.join(script_directory, 'default_files_config', 'qubit_spectroscopy', f'{qubit}.txt')
            qubit_default_files_config_filename = os.path.join(self.root_directory, 'default_files_config', 'qubit_spectroscopy', f'{qubit}.txt')


            # qubit_spec_data_filename = os.path.join(script_directory, r'data\qubit_spectroscopy', qubit)
            qubit_spec_data_filename = os.path.join(self.qubit_spectroscopy_directory, qubit, qubit)
            qubit_spec_data_filtered_directory = os.path.join(self.qubit_spectroscopy_directory, qubit)
            qubit_spec_fit_parameters_filename = os.path.join(self.qubit_spectroscopy_directory, qubit, f'{qubit}_fit_parameters.txt')


            if qubit == 'Q4':
                measurement_type = 'phase'
            else:
                measurement_type = 'amp'

            ### load files widget
            print(f'Creating load files widget for {qubit}')
            load_files_widget = LoadFilesWidget(qubit, self.measurements_directory, qubit_default_files_config_filename, measurement_type=measurement_type)
            # load_files_widget.load_files()

            ### peak fit widget
            print(f'Creating peak fit widget for {qubit}')
            peak_fit_widget = PeakFitWidget(qubit, load_files_widget, data_filepath=qubit_spec_data_filename)
            peak_fit_widget.peaks_fit.connect(load_files_widget.plot_files)
            # peak_fit_widget.fit_peaks()


            ### manual filter data widget
            print(f'Creating manual filter data widget for {qubit}')
            manual_filter_data_widget = ManualFilterDataWidget(qubit, peak_fit_widget.get_voltages, peak_fit_widget.get_frequencies, data_directory=qubit_spec_data_filtered_directory)
            peak_fit_widget.peaks_fit.connect(manual_filter_data_widget.update_data)

            ### fit transmon widget
            print(f'Creating fit transmon widget for {qubit}')
            fit_transmon_widget = FitTransmonWidget(qubit, manual_filter_data_widget.get_filtered_x_data, manual_filter_data_widget.get_filtered_y_data, data_filepath=qubit_spec_fit_parameters_filename)



            load_files_widget.run_startup()
            peak_fit_widget.run_startup()
            manual_filter_data_widget.run_startup()
            fit_transmon_widget.run_startup()

            # Create QGroupBox for Load Files section
            load_files_group_box = QGroupBox("Load Files")
            load_files_layout = QVBoxLayout()
            load_files_layout.addWidget(load_files_widget)
            load_files_group_box.setLayout(load_files_layout)

            # Create QGroupBox for Peak Fit section
            peak_fit_group_box = QGroupBox("Peak Fit")
            peak_fit_layout = QVBoxLayout()
            peak_fit_layout.addWidget(peak_fit_widget)
            peak_fit_group_box.setLayout(peak_fit_layout)

            # Create QGroupBox for Manual Filter Data section
            manual_filter_group_box = QGroupBox("Manually Filter Data")
            manual_filter_layout = QVBoxLayout()
            manual_filter_layout.addWidget(manual_filter_data_widget)
            manual_filter_group_box.setLayout(manual_filter_layout)

            # Create QGroupBox for Fit Transmon section
            fit_transmon_group_box = QGroupBox("Fit Transmon Model")
            fit_transmon_layout = QVBoxLayout()
            fit_transmon_layout.addWidget(fit_transmon_widget)
            fit_transmon_group_box.setLayout(fit_transmon_layout)

            # Add the group boxes to the tab content layout
            tab_content_layout.addWidget(load_files_group_box)
            tab_content_layout.addWidget(peak_fit_group_box)
            tab_content_layout.addWidget(manual_filter_group_box)
            tab_content_layout.addWidget(fit_transmon_group_box)

            tab.setLayout(QVBoxLayout())
            tab.layout().addWidget(scroll_area)

        main_layout.addWidget(self.tab_widget)
