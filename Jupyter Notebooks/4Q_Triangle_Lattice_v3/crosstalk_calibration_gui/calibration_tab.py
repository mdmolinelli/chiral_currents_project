import numpy as np
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QButtonGroup, QPushButton
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.optimize import minimize


class CalibrationTab(QWidget):
    def __init__(self, spectroscopy_tab):
        super().__init__()

        self.spectroscopy_tab = spectroscopy_tab

        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        inputs_layout = QVBoxLayout()

        # Horizontal group of radio buttons
        radio_buttons_row = QHBoxLayout()
        labels = ['Q2', 'Q3', 'Q4', 'C12', 'C13', 'C23', 'C24', 'C34']
        self.radio_buttons = []
        self.radio_button_group = QButtonGroup()
        for label in labels:
            radio_button = QRadioButton(label)
            self.radio_buttons.append(radio_button)
            self.radio_button_group.addButton(radio_button)
            radio_buttons_row.addWidget(radio_button)

        inputs_layout.addLayout(radio_buttons_row)

        # Expected flux with a float restricted LineEdit
        expected_flux_row = QHBoxLayout()
        expected_flux_label = QLabel('Expected flux:')
        self.expected_flux_input = QLineEdit()
        self.expected_flux_input.setValidator(QDoubleValidator())
        self.expected_flux_input.setText('0')
        expected_flux_row.addWidget(expected_flux_label)
        expected_flux_row.addWidget(self.expected_flux_input)

        inputs_layout.addLayout(expected_flux_row)

        # Number to fit with an integer restricted LineEdit
        number_to_fit_row = QHBoxLayout()
        number_to_fit_label = QLabel('Number to fit:')
        self.number_to_fit_input = QLineEdit()
        self.number_to_fit_input.setValidator(QIntValidator())
        self.number_to_fit_input.setText('10')
        number_to_fit_row.addWidget(number_to_fit_label)
        number_to_fit_row.addWidget(self.number_to_fit_input)

        inputs_layout.addLayout(number_to_fit_row)

        # Fit offset vector with yes/no radio buttons
        fit_offset_vector_row = QHBoxLayout()
        fit_offset_vector_label = QLabel('Fit offset vector:')
        fit_offset_vector_row.addWidget(fit_offset_vector_label)

        self.fit_offset_vector_yes = QRadioButton('Yes')
        self.fit_offset_vector_no = QRadioButton('No')
        self.fit_offset_vector_yes.setChecked(True)  # Default to 'Yes'

        fit_offset_vector_button_group = QButtonGroup()
        fit_offset_vector_button_group.addButton(self.fit_offset_vector_yes)
        fit_offset_vector_button_group.addButton(self.fit_offset_vector_no)

        fit_offset_vector_row.addWidget(self.fit_offset_vector_yes)
        fit_offset_vector_row.addWidget(self.fit_offset_vector_no)

        inputs_layout.addLayout(fit_offset_vector_row)

        # calibrate button
        calibrate_button = QPushButton('calibrate')
        calibrate_button.clicked.connect(self.calibrate)
        inputs_layout.addWidget(calibrate_button)

        layout.addLayout(inputs_layout)


        ### fit plot

        fig, ax = plt.subplots()
        fit_canvas = FigureCanvas(fig)
        # canvas.setMinimumSize(*min_size)
        # canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(fit_canvas)

    def calibrate(self):

        qubit = 'Q2'
        coupler = 'C12'

        # voltages = self.spectroscopy_tab.qubits_tab.qubit_to_voltages[qubit]['calibration']
        # fluxes = self.spectroscopy_tab.qubits_tab.qubit_to_calibration_fluxes[qubit]

        voltages = self.spectroscopy_tab.couplers_tab.qubit_coupler_to_voltages[coupler]['calibration']
        fluxes = self.spectroscopy_tab.couplers_tab.qubit_to_calibration_fluxes[coupler]

        print(f'voltages: {np.round(voltages, 2)}')
        print(f'fluxes: {np.round(fluxes, 2)}')

        pass

def error_function(crosstalk_row, voltages, fluxes, flux_quanta, offset_value, index, value_to_add=1, fit_offset_vector=False):
    crosstalk_row = np.insert(crosstalk_row, index, value_to_add)

    if fit_offset_vector:
        offset_value = crosstalk_row[-1]
        crosstalk_row = np.delete(crosstalk_row, -1)
    expected_fluxes = []
    for i, v in enumerate(voltages):
        flux_value = crosstalk_row.dot(np.array(v))
        expected_fluxes.append((flux_value - offset_value) / flux_quanta)
    error_value = np.sum((np.array(expected_fluxes) - np.array(fluxes)) ** 2)
    print(error_value)
    return (error_value)

def minimize_crosstalk_error(fluxes, voltages, crosstalk_guess, offset_value, flux_quanta, index, fit_offset_vector = False):
    result = minimize(lambda x: error_function(x, voltages, fluxes, flux_quanta, offset_value, index,
                                     fit_offset_vector=fit_offset_vector), crosstalk_guess,
                   method='Nelder-Mead', tol = 0.000001)
    return result
