import sys
import os
import itertools

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox, QScrollArea
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from custom_widgets.load_files_widget import LoadFilesWidget
from custom_widgets.manual_filter_data_widget import ManualFilterDataWidget
from custom_widgets.peak_fit_widget import PeakFitWidgetAvoidedCrossings
from custom_widgets.fit_transmon import FitTransmonWidget

class AvoidedCrossingMeasurementsTab(QWidget):
    qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']



    def __init__(self, root_directory, measurements_directory):
        super().__init__()
        self.root_directory = root_directory

        self.qubit_avoided_crossings_directory = os.path.join(self.root_directory, 'data/qubit_avoided_crossings')
        # self.default_config_directory = os.path.join(self.root_directory, 'data/qubit_avoided_crossings')

        self.measurements_directory = measurements_directory
        self.initUI()

    def initUI(self):
        self.tab_widget = QTabWidget()
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # for qubit_1, qubit_2 in self.get_all_qubit_pairs():
        for qubit_1, qubit_2 in [('Q1', 'Q3')]:

            if not self.is_adjacent(qubit_1, qubit_2):
                continue

            qubit_pair_label = f'{qubit_1}-{qubit_2}'

            tab = QWidget()
            self.tab_widget.addTab(tab, qubit_pair_label)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)

            tab_content_widget = QWidget()
            tab_content_layout = QVBoxLayout(tab_content_widget)
            scroll_area.setWidget(tab_content_widget)

            qubit_default_files_config_filename = os.path.join(self.root_directory, 'default_files_config', 'avoided_crossings', f'{qubit_pair_label}.txt')
            qubit_avoided_crossings_data_filename = os.path.join(self.qubit_avoided_crossings_directory, qubit_pair_label, qubit_pair_label)
            qubit_spec_data_filtered_directory = os.path.join(self.qubit_avoided_crossings_directory, qubit_pair_label)
            qubit_avoided_crossings_data_filtered_filename = os.path.join(self.qubit_avoided_crossings_directory, qubit_pair_label, f'{qubit_pair_label}_filtered')
            qubit_avoided_crossings_fit_parameters_filename = os.path.join(self.qubit_avoided_crossings_directory, qubit_pair_label, f'{qubit_pair_label}_fit_parameters')

            # create label args for LoadFilesWidget
            if self.is_tunable_coupling(qubit_1, qubit_2):
                label_args = (('Qubit Frequency', float), ('Coupler Voltage', float))
            elif self.is_adjacent(qubit_1, qubit_2):
                label_args = (('Qubit Frequency', float),)

            print(f'creating widgets for {qubit_pair_label}')
            load_files_widget = LoadFilesWidget(qubit_pair_label, self.measurements_directory, qubit_default_files_config_filename, file_label_args=label_args)
            peak_fit_widget = PeakFitWidgetAvoidedCrossings((qubit_1, qubit_2), load_files_widget, data_filepath=qubit_avoided_crossings_data_filename)
            load_files_widget.selected_file_changed.connect(peak_fit_widget.selected_file_changed_handler)
            
            peak_fit_widget.peaks_fit.connect(lambda fit_voltages, fit_frequencies: load_files_widget.update_plot_parameters(fit_voltages=fit_voltages, fit_frequencies=fit_frequencies))   
            peak_fit_widget.separator_parameters_changed.connect(lambda middle_index, separator_slope: load_files_widget.update_plot_parameters(middle_frequency_index=middle_index, separator_slope=separator_slope))

            manual_filter_data_widget = ManualFilterDataWidget((qubit_1, qubit_2), peak_fit_widget.get_voltages, peak_fit_widget.get_frequencies, data_directory=qubit_spec_data_filtered_directory)
            # peak_fit_widget.peaks_fit.connect(manual_filter_data_widget.update_data)

            # Placeholder for Fit Avoided Crossing Widget
            fit_avoided_crossing_widget = QWidget()  # Replace with actual widget implementation

            load_files_widget.run_startup()
            peak_fit_widget.run_startup()
            # manual_filter_data_widget.run_startup()
            # fit_avoided_crossing_widget.run_startup()  # Uncomment when implemented

            load_files_group_box = QGroupBox("Load Files")
            load_files_layout = QVBoxLayout()
            load_files_layout.addWidget(load_files_widget)
            load_files_group_box.setLayout(load_files_layout)

            peak_fit_group_box = QGroupBox("Peak Fit")
            peak_fit_layout = QVBoxLayout()
            peak_fit_layout.addWidget(peak_fit_widget)
            peak_fit_group_box.setLayout(peak_fit_layout)

            manual_filter_group_box = QGroupBox("Manually Filter Data")
            manual_filter_layout = QVBoxLayout()
            manual_filter_layout.addWidget(manual_filter_data_widget)
            manual_filter_group_box.setLayout(manual_filter_layout)

            # fit_avoided_crossing_group_box = QGroupBox("Fit Avoided Crossing")
            # fit_avoided_crossing_layout = QVBoxLayout()
            # fit_avoided_crossing_layout.addWidget(fit_avoided_crossing_widget)
            # fit_avoided_crossing_group_box.setLayout(fit_avoided_crossing_layout)

            tab_content_layout.addWidget(load_files_group_box)
            tab_content_layout.addWidget(peak_fit_group_box)
            tab_content_layout.addWidget(manual_filter_group_box)
            # tab_content_layout.addWidget(fit_avoided_crossing_group_box)

            tab.setLayout(QVBoxLayout())
            tab.layout().addWidget(scroll_area)

        main_layout.addWidget(self.tab_widget)

    def get_all_qubit_pairs(self):
        return itertools.product(self.qubits, self.qubits) 
    
    def is_adjacent(self, qubit_1, qubit_2):

        qubit_1_index = int(qubit_1[1])
        qubit_2_index = int(qubit_2[1])

        if 0 < qubit_2_index - qubit_1_index <= 2:
            return True

        return False
        
    def is_tunable_coupling(self, qubit_1, qubit_2):

        if not self.is_adjacent(qubit_1, qubit_2):
            return False

        qubit_1_index = int(qubit_1[1])
        qubit_2_index = int(qubit_2[1])
        
        if qubit_2_index - qubit_1_index == 2:
            return True
        
        return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    root_directory = r'C:\path\to\your\root\directory'
    widget = AvoidedCrossingMeasurementsTab(root_directory)
    widget.show()
    sys.exit(app.exec_())