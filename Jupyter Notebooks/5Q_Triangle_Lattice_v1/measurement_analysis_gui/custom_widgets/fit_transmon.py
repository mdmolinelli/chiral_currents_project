import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QApplication, QLabel)
from PyQt5.QtCore import pyqtSignal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from scipy.optimize import curve_fit

from util.model_functions import transmon_model_fit
import csv

class FitTransmonWidget(QWidget):
    data_fit = pyqtSignal(np.ndarray, np.ndarray)
    data_saved = pyqtSignal(str)

    def __init__(self, qubit, voltage_getter, frequency_getter, data_filepath):
        super().__init__()

        self.qubit = qubit
        self.voltage_getter = voltage_getter
        self.frequency_getter = frequency_getter
        self.data_filepath = data_filepath

        self.initUI()

        

    def initUI(self):
        self.setFixedSize(800, 600)  # Set the desired width and height

        # Initialize the plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Buttons
        self.fit_data_btn = QPushButton('Fit Data', self)
        self.fit_data_btn.clicked.connect(self.fit_data)

        self.save_data_btn = QPushButton('Save Data', self)
        self.save_data_btn.clicked.connect(self.save_data)

        # Layout
        main_layout = QHBoxLayout(self)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.fit_data_btn)
        button_layout.addWidget(self.save_data_btn)

        layout.addLayout(button_layout)


        popt_layout = QVBoxLayout()
        self.popt_label = QLabel('Fit Parameters:')
        popt_layout.addWidget(self.popt_label)

        layout.addLayout(popt_layout)

        main_layout.addLayout(layout)


        self.setLayout(main_layout)

    def run_startup(self):
        # Load data if file exists
        if os.path.exists(self.data_filepath):
            self.load_data()

            self.plot_data()

    def load_data(self):
        """Load data from the file and plot it."""
        try:
            with open(self.data_filepath, 'r') as file:
                reader = csv.reader(file)
                data = []
                for row in reader:
                    data.append(row)
                transmon_popt = np.array(data[1], dtype=float)
                self.update_transmon_popt(transmon_popt)
        except Exception as e:
            print(f"Error loading data: {e}")

    def plot_data(self):
        """Plot the data on the canvas."""
        self.ax.clear()
        self.ax.plot(self.get_voltages(), self.get_frequencies(), 'o', label='data')

        self.ax.plot(self.get_fit_voltages(), transmon_model_fit(self.get_fit_voltages(), *self.transmon_popt), label='fit')
        
        self.ax.set_title(f'{self.qubit} Spectroscopy vs flux: transmon model')
        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Frequency (GHz)')
        self.ax.legend()
        self.canvas.draw()

    def fit_data(self):
        """Fit the data (to be implemented by the user)."""
        transmon_initial_guess = [-0.5, 30, 1.2, 0.2061, 0.6]
        transmon_bounds = ((-np.inf, 0, 0, 0.206, 0), (np.inf, np.inf, np.inf, 0.207, 1))
        

        try:
            transmon_popt, self.transmon_pcov = curve_fit(transmon_model_fit, self.get_voltages(), self.get_frequencies(), p0=transmon_initial_guess, bounds=transmon_bounds)
            self.update_transmon_popt(transmon_popt)
        except Exception as e:
            # if fit failed, plot data and guess
            self.ax.clear()
            self.ax.plot(self.get_voltages(), self.get_frequencies(), 'o-', label='Data')
            self.ax.plot(self.get_fit_voltages(), transmon_model_fit(self.get_fit_voltages(), *transmon_initial_guess), label='guess')
            
            self.ax.set_title(f'{self.qubit} Spectroscopy vs flux: transmon model')
            self.ax.set_xlabel('Voltage (V)')
            self.ax.set_ylabel('Frequency (GHz)')
            self.ax.legend()
            self.canvas.draw()

        else:
            self.transmon_err = np.sqrt(np.diag(self.transmon_pcov))
            self.data_fit.emit(self.get_voltages(), self.get_frequencies())

            self.plot_data()

    def save_data(self):
        """Save the fit data to the file."""
        try:
            with open(self.data_filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x0', 'a', 'b', 'c', 'd'])
                writer.writerow(self.transmon_popt)

            self.data_saved.emit(self.data_filepath)
            print(f"Data saved to {self.data_filepath}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def get_voltages(self):
        return self.voltage_getter()
    
    def get_frequencies(self):
        return self.frequency_getter()
    
    def get_fit_voltages(self):
        voltages = self.get_voltages()
        if len(voltages) > 0:
            print(voltages)
            return np.linspace(voltages[0], voltages[-1], 1001)
        else:
            return []
        
    def update_transmon_popt(self, popt):
        self.transmon_popt = popt
        self.popt_label.setText(f'Fit Parameters: {popt}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    voltages = np.array([0, 1, 2, 3, 4, 5])
    frequencies = np.array([10, 20, 30, 40, 50, 60])
    data_filepath = 'transmon_data.csv'
    widget = FitTransmonWidget(voltages, frequencies, data_filepath)
    widget.show()
    sys.exit(app.exec_())