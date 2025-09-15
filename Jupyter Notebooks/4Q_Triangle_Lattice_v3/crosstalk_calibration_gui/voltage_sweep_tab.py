import csv
import datetime
import os

import numpy as np
import random

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox, \
    QScrollArea, QTabWidget, QCheckBox

from crosstalk_matrix_tab import CrosstalkMatrixTab
from crosstalk_compensation_optimization import CrosstalkCompensationOptimization


class RandomVoltagesTab(QWidget):
    def __init__(self):
        super().__init__()
        self.filepath_display = None

        self.initUI()

        self.random_fluxes = None

        self.crosstalk_matrix = None
        self.crosstalk_inverse_matrix = None
        self.crosstalk_offset_vector = None

        self.crosstalk_compensation_optimization = None

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.createGenerateFluxesGroup())
        layout.addWidget(self.createConvertVoltagesGroup())
        layout.addWidget(self.createExportVoltagesGroup())
        self.setLayout(layout)

    def createGenerateFluxesGroup(self):
        group = QGroupBox("Generate Random Fluxes")
        layout = QGridLayout()
        layout.addWidget(QLabel(""), 0, 0)
        labels = ["Q2", "Q3", "Q4", "C12", "C13", "C23", "C24", "C34"]
        for i, label in enumerate(labels, start=1):
            layout.addWidget(QLabel(label), 0, i)

        layout.addWidget(QLabel("number of random samples:"), 1, 0)
        num_samples_line_edit = QLineEdit()
        num_samples_line_edit.setValidator(QIntValidator())
        num_samples_line_edit.setText('40')
        layout.addWidget(num_samples_line_edit, 1, 1)

        layout.addWidget(QLabel("fixed channels:"), 2, 0)
        layout.addWidget(QLabel("flux points:"), 3, 0)
        layout.addWidget(QLabel("lower bound:"), 4, 0)
        layout.addWidget(QLabel("upper bound:"), 5, 0)

        self.fixed_channels_checkboxes = []
        self.flux_points_line_edits = []
        self.lower_bounds_line_edits = []
        self.upper_bounds_line_edits = []

        validator = QDoubleValidator()

        for i in range(8):
            fixed_channel = QCheckBox()
            layout.addWidget(fixed_channel, 2, i + 1)
            self.fixed_channels_checkboxes.append(fixed_channel)

            flux_point = QLineEdit()
            flux_point.setValidator(validator)
            flux_point.setText("0")
            layout.addWidget(flux_point, 3, i+1)
            self.flux_points_line_edits.append(flux_point)

            lower_bound = QLineEdit()
            lower_bound.setValidator(validator)
            lower_bound.setText("0.3")
            layout.addWidget(lower_bound, 4, i+1)
            self.lower_bounds_line_edits.append(lower_bound)

            upper_bound = QLineEdit()
            upper_bound.setValidator(validator)
            upper_bound.setText("0.7")
            layout.addWidget(upper_bound, 5, i+1)
            self.upper_bounds_line_edits.append(upper_bound)

        generate_random_fluxes_button = QPushButton("Generate Random Fluxes")
        # connect button clicked with lambda function with fixed channels, flux points, and bounds specified
        generate_random_fluxes_button.clicked.connect(lambda: self.generate_random_fluxes(
            int(num_samples_line_edit.text()),
            np.where([checkbox.checkState() == Qt.Checked for checkbox in self.fixed_channels_checkboxes])[0],
            [float(line_edit.text()) for line_edit in self.flux_points_line_edits],
            [(float(self.lower_bounds_line_edits[k].text()), float(self.upper_bounds_line_edits[k].text())) for k in range(len(self.lower_bounds_line_edits))],
        ))

        layout.addWidget(generate_random_fluxes_button, 6, 0, 1, 9)

        # Scrollable text box
        self.generated_fluxes_display = QTextEdit()
        self.generated_fluxes_display.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.generated_fluxes_display)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, 7, 0, 1, 9)

        group.setLayout(layout)
        return group

    def createConvertVoltagesGroup(self):
        group = QGroupBox("Convert to Voltages")
        layout = QGridLayout()

        layout.addWidget(QLabel("single channel threshold:"), 0, 0)
        single_channel_threshold_line_edit = QLineEdit()
        single_channel_threshold_line_edit.setValidator(QDoubleValidator())
        single_channel_threshold_line_edit.setText('2')
        layout.addWidget(single_channel_threshold_line_edit, 0, 1)

        convert_button = QPushButton("Convert to Voltages")
        convert_button.clicked.connect(lambda: self.convert_to_random_voltages(
            self.random_fluxes,
            self.fixed_channels,
            float(single_channel_threshold_line_edit.text())
        ))
        layout.addWidget(convert_button, 1, 0, 1, 2)

        # Scrollable text box
        self.converted_voltages_display = QTextEdit()
        self.converted_voltages_display.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.converted_voltages_display)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, 2, 0, 1, 2)

        group.setLayout(layout)
        return group

    def createExportVoltagesGroup(self):
        group = QGroupBox("Export Voltages")
        layout = QVBoxLayout()

        export_button = QPushButton("Export Voltages")
        export_button.clicked.connect(self.export_random_voltages)
        layout.addWidget(export_button)

        self.filepath_display = QLineEdit()
        self.filepath_display.setReadOnly(True)
        layout.addWidget(self.filepath_display)

        group.setLayout(layout)
        return group


    def generate_random_fluxes(self, num_random_samples, fixed_channels, flux_points, bounds):

        random_fluxes = np.zeros((num_random_samples, len(flux_points)))

        for i in range(num_random_samples):
            for j in range(len(flux_points)):
                # if fixed channel, use point in flux points
                if j in fixed_channels:
                    random_fluxes[i, j] = flux_points[j]
                    continue

                flux_range = bounds[j]

                if len(flux_range) > 0:
                    random_flux = random.uniform(*flux_range)
                else:
                    random_flux = flux_points[j]
                random_fluxes[i, j] = random_flux

        # lock in fixed indices, so that changing checkboxes doesn't matter anymore until this button is clicked again
        self.fixed_channels = fixed_channels

        self.random_fluxes = random_fluxes
        self.generated_fluxes_display.setText(np.array_str(np.round(random_fluxes, 2)))

    def convert_to_random_voltages(self, random_fluxes, fixed_channels, single_channel_threshold):

        random_voltages = []

        for i in range(len(random_fluxes)):

            fluxes = random_fluxes[i, :]
            optimal_adjustments = self.crosstalk_compensation_optimization.integer_programming_get_combination(
                fluxes,
                single_channel_threshold=single_channel_threshold,
                fixed_indices=fixed_channels
            )

            optimal_fluxes = fluxes + optimal_adjustments

            voltages = self.crosstalk_compensation_optimization.flux_to_voltage(optimal_fluxes)

            random_voltages.append(voltages)

        self.random_voltages = np.array(random_voltages)

        self.converted_voltages_display.setText(np.array_str(np.round(random_voltages, 2)))

    def set_crosstalk_matrix(self, matrix_label, matrix):
        if matrix_label == CrosstalkMatrixTab.crosstalk_matrix_label:
            self.crosstalk_matrix = matrix
        elif matrix_label == CrosstalkMatrixTab.crosstalk_inverse_matrix_label:
            self.crosstalk_inverse_matrix = matrix
        elif matrix_label == CrosstalkMatrixTab.crosstalk_offset_vector_label:
            self.crosstalk_offset_vector = matrix


        self.crosstalk_compensation_optimization = CrosstalkCompensationOptimization(
            self.crosstalk_matrix,
            self.crosstalk_inverse_matrix,
            self.crosstalk_offset_vector
        )

    def export_random_voltages(self):
        now = datetime.datetime.now()
        current_datetime_string = now.strftime('%Y-%m%d_%H%M')

        directory_name = f'random_voltages_{now.strftime("%Y-%m%d")}'

        random_voltages_directory = r'V:\QSimMeasurements\Measurements\4Q_Triangle_Lattice\voltage_sweeps\{}'.format(
            directory_name)

        random_voltages_directory = 'random_voltages_test'
        os.makedirs(random_voltages_directory, exist_ok=True)

        filepath = os.path.join(random_voltages_directory, 'random_voltages_{}.csv'.format(current_datetime_string))

        print(f'saving to: {filepath}')

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            for i in range(self.random_voltages.shape[0]):
                writer.writerow(self.random_voltages[i, :])

        self.filepath_display.setText(filepath)

class VoltageSweepSubTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.createGenerateSweepGroup())
        layout.addWidget(self.createExportVoltagesGroup())
        self.setLayout(layout)

    def createGenerateSweepGroup(self):
        group = QGroupBox("Generate Voltage Sweep")
        layout = QVBoxLayout()
        generate_button = QPushButton("Generate Voltage Sweep")
        layout.addWidget(generate_button)
        group.setLayout(layout)
        return group

    def createExportVoltagesGroup(self):
        group = QGroupBox("Export Voltages")
        layout = QVBoxLayout()
        export_button = QPushButton("Export Voltages")
        layout.addWidget(export_button)
        group.setLayout(layout)
        return group


class VoltageSweepTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        tabs = QTabWidget()

        self.random_voltages_tab = RandomVoltagesTab()
        self.voltage_sweep_subtab = VoltageSweepSubTab()

        tabs.addTab(self.random_voltages_tab, "Random Voltages")
        tabs.addTab(self.voltage_sweep_subtab, "Voltage Sweep")
        layout.addWidget(tabs)
        self.setLayout(layout)
