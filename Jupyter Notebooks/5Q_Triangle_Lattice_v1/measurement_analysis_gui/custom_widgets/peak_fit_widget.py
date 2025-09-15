import csv
import sys
import os
import numpy as np


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox,
                             QFileDialog, QApplication, QHBoxLayout, QLineEdit, QGridLayout)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from util.data_util import get_center_frequencies, get_avoided_crossing_frequencies
from custom_widgets.custom_widget import CustomWidget


class PeakFitWidget(QWidget):

    peaks_fit = pyqtSignal(np.ndarray, np.ndarray)
    

    def __init__(self, qubit, load_files_widget, data_filepath):
        super().__init__()
        # Rest of the code...

        # Initialize the class variables

        self.qubit = qubit
        self.load_files_widget = load_files_widget

        self.voltages = []
        self.frequencies = []
        self.frequency_errors = []

        self.data_filepath = data_filepath

        self.show_fits = False  # Boolean to track the checkbox status
        self.initUI()

        

    def initUI(self):
        # Set the fixed size of the widget
        self.setFixedSize(1200, 600)  # Set the desired width and height


        # Main layout
        main_layout = QVBoxLayout(self)

        

        self.create_fit_button(main_layout)

        # Checkbox for "Show Fits"
        checkbox_layout = QHBoxLayout()
        self.show_fits_checkbox = QCheckBox(self)
        self.show_fits_checkbox.stateChanged.connect(self.show_fits_toggled)

        checkbox_layout.addWidget(QLabel('Show Fits:', self))
        checkbox_layout.addWidget(self.show_fits_checkbox)

        main_layout.addLayout(checkbox_layout)

        # Create matplotlib canvas for plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def create_fit_button(self, layout):
        # Button for fitting peaks
        fit_peaks_btn = QPushButton('Fit Peaks', self)
        fit_peaks_btn.clicked.connect(self.fit_peaks)
        layout.addWidget(fit_peaks_btn)

    def run_startup(self):
        # If file exists, load
        if self.data_filepath and os.path.exists(self.get_data_filepath()):
            self.load_data()
        else:
            self.fit_peaks()

    def load_data(self):
        """Load the data from a CSV file."""

        try:
            data_filepath = self.get_data_filepath()
            with open(data_filepath, 'r') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader)  # Skip the header row, assuming the file has headers

                for row in csv_reader:
                    self.voltages.append(float(row[0]))
                    self.frequencies.append(float(row[1]))

        except Exception as e:
            print(f"Error loading data from {self.data_filepath}: {e}")

        self.peaks_fit.emit(np.array(self.voltages), np.array(self.frequencies))

        self.plot_data()

    def save_data(self):
        """Save the data to a CSV file."""
        try:
            os.makedirs(os.path.dirname(self.data_filepath), exist_ok=True)
            data_filepath = self.get_data_filepath()
            print(f"Saving data to {data_filepath}")
            with open(data_filepath, 'w', newline='') as file:
                csv_writer = csv.writer(file)

                # Write headers
                csv_writer.writerow(['Voltage', 'Frequency'])

                # Write data rows
                voltages = self.get_voltages()
                frequencies = self.get_frequencies()
                for i in range(len(voltages)):
                    row = [voltages[i], frequencies[i]]
                    csv_writer.writerow(row)

        except Exception as e:
            print(f"Error saving data to {self.data_filepath}: {e}")

    def plot_data(self):
        """Plot the voltage, frequency, and transmission data."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)


        voltages = self.get_voltages()
        frequencies = self.get_frequencies()

        if voltages is not None and frequencies is not None:
            ax.plot(voltages, frequencies, 'o')

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('Frequency vs Voltage')
        self.canvas.draw()

    def fit_peaks(self):

        voltages = []
        frequencies = []
        frequency_errors = []

        for key in self.load_files_widget.voltage_data_all:

            voltage_data = self.load_files_widget.voltage_data_all[key]
            frequency_data = self.load_files_widget.frequency_data_all[key]
            transmission_data = self.load_files_widget.transmission_data_all[key]

            file_voltages, file_frequencies, error = get_center_frequencies(voltage_data, frequency_data,
                                                                            transmission_data,
                                                                            plot_fits=self.show_fits)

            voltages.extend(file_voltages)
            frequencies.extend(file_frequencies)
            frequency_errors.extend(error)

        voltages = np.array(voltages)
        frequencies = np.array(frequencies)
        frequency_errors = np.array(frequency_errors)

        sorted_indices = voltages.argsort()
        voltages = voltages[sorted_indices]
        frequencies = frequencies[sorted_indices]
        frequency_errors = frequency_errors[sorted_indices]

        self.voltages = voltages
        self.frequencies = frequencies
        self.frequency_errors = frequency_errors

        self.save_data()
        self.plot_data()

        self.peaks_fit.emit(self.voltages, self.frequencies)

    def show_fits_toggled(self):
        """Handle the toggling of the 'Show Fits' checkbox."""
        self.show_fits = self.show_fits_checkbox.isChecked()
        print(f"Show Fits: {self.show_fits}")
        # Depending on self.show_fits, you can update the plot to show the fits

    def get_data_filepath(self):
        return self.data_filepath + '.csv'
    
    def get_voltages(self):
        return self.voltages
    
    def get_frequencies(self):
        return self.frequencies
    
    def get_frequency_errors(self):
        return self.frequency_errors

class PeakFitWidgetAvoidedCrossings(PeakFitWidget):

    # Signal for when the line that separates peaks in the avoided crossing is changed. Signal is (middle index, slope)
    separator_parameters_changed = pyqtSignal(int, float)

    def __init__(self, qubit_pair, load_files_widget, data_filepath):
        super().__init__(qubit_pair, load_files_widget, data_filepath)

        self.selected_file_labels = None

        self.voltages = {}
        self.frequencies = {}
        self.frequency_errors = {}

        self.middle_index_dict = {}
        self.separator_slope_dict = {}

    def create_fit_button(self, layout):
        # Button for fitting peaks
        fit_peaks_btn = QPushButton('Fit Peaks', self)
        fit_peaks_btn.clicked.connect(self.fit_peaks)
        layout.addWidget(fit_peaks_btn)

        separator_parameters_layout = QGridLayout()

        separator_parameters_layout.addWidget(QLabel('Middle Frequency Index:', self), 0, 0)
        self.middle_index_input = QLineEdit(self)
        self.middle_index_input.setText('0')
        self.middle_index_input.setValidator(QIntValidator())


        separator_parameters_layout.addWidget(self.middle_index_input, 0, 1)

        self.separator_slope_input = QLineEdit(self)
        self.separator_slope_input.setText('0')
        self.separator_slope_input.setValidator(QDoubleValidator())

        self.middle_index_input.editingFinished.connect(lambda: self.separator_parameters_changed.emit(int(self.middle_index_input.text()), 
                                                                                                 float(self.separator_slope_input.text())))
        self.separator_slope_input.editingFinished.connect(lambda: self.separator_parameters_changed.emit(int(self.middle_index_input.text()), 
                                                                                                 float(self.separator_slope_input.text())))  
        
        separator_parameters_layout.addWidget(QLabel('Separator Slope:', self), 1, 0)
        separator_parameters_layout.addWidget(self.separator_slope_input, 1, 1) 


        layout.addLayout(separator_parameters_layout)

    def fit_peaks(self):

        if self.selected_file_labels is None:
            return

        voltages = {}
        frequencies = {}
        frequency_errors = {}

        start_index = 5
        frequency_index_span = 25

        voltage_data = self.load_files_widget.voltage_data_all[self.selected_file_labels]
        frequency_data = self.load_files_widget.frequency_data_all[self.selected_file_labels]
        transmission_data = self.load_files_widget.transmission_data_all[self.selected_file_labels]

        file_voltages, file_frequencies, error = get_avoided_crossing_frequencies(voltage_data, frequency_data,
                                                                                  transmission_data, middle_frequency_index=int(self.middle_index_input.text()),
                                                                                  separator_slope=float(self.separator_slope_input.text()),
                                                                                  start_index=start_index, frequency_index_span=frequency_index_span,
                                                                                  plot_fits=self.show_fits)

        voltages = np.array(file_voltages)
        frequencies = np.array(file_frequencies)
        frequency_errors = np.array(error)

        # voltages is a list with 2 elements, so we need to sort the data based on the first element


        sorted_indices = voltages[0].argsort()

        for i in range(len(voltages)):
            voltages[i] = voltages[i][sorted_indices]
            frequencies[i] = frequencies[i][sorted_indices]
            frequency_errors[i] = frequency_errors[i][sorted_indices]

        self.voltages[self.selected_file_labels] = voltages
        self.frequencies[self.selected_file_labels] = frequencies
        self.frequency_errors[self.selected_file_labels] = frequency_errors

        self.save_data()
        self.plot_data()

        self.peaks_fit.emit(voltages, frequencies)

    def plot_data(self):
        """Plot the voltage, frequency, and transmission data."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)


        voltages = self.get_voltages()
        frequencies = self.get_frequencies()

        if voltages is not None and frequencies is not None:
            ax.plot(voltages[0], frequencies[0], 'o')
            ax.plot(voltages[1], frequencies[1], 'o')

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('Frequency vs Voltage')
        self.canvas.draw()

    def save_data(self):
        """Save the data to a CSV file."""
        try:
            os.makedirs(os.path.dirname(self.data_filepath), exist_ok=True)
            data_filepath = self.get_data_filepath()
            print(f"Saving data to {data_filepath}")
            with open(data_filepath, 'w', newline='') as file:
                csv_writer = csv.writer(file)

                # Write headers
                csv_writer.writerow(['Voltage 1', 'Frequency 1', 'Voltage 2', 'Frequency 2, Middle Frequency Index, Separator Slope'])

                # Write data rows
                voltages = self.get_voltages()
                frequencies = self.get_frequencies()
                for i in range(len(voltages[0])):
                    if i == 0:
                        row = [voltages[0][i], frequencies[0][i], voltages[1][i], frequencies[1][i], int(self.middle_index_input.text()), float(self.separator_slope_input.text())]
                    else:
                        row = [voltages[0][i], frequencies[0][i], voltages[1][i], frequencies[1][i]]
                    csv_writer.writerow(row)

        except Exception as e:
            print(f"Error saving data to {self.data_filepath}: {e}")

    def load_data(self):
        """Load the data from a CSV file."""

        try:

            voltages_1 = []
            frequencies_1 = []
            voltages_2 = []
            frequencies_2 = []

            data_filepath = self.get_data_filepath()
            with open(data_filepath, 'r') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader)  # Skip the header row, assuming the file has headers

                row_1 = next(csv_reader)
                voltages_1.append(float(row_1[0]))
                frequencies_1.append(float(row_1[1]))
                voltages_2.append(float(row_1[2]))
                frequencies_2.append(float(row_1[3]))

                if len(row_1) > 5:
                    self.middle_index_input.setText(row_1[4])
                    self.middle_index_dict[self.selected_file_labels] = int(row_1[4])
                    # self.middle_index_input.editingFinished.emit()

                    self.separator_slope_input.setText(row_1[5])
                    self.separator_slope_dict[self.selected_file_labels] = float(row_1[5])
                    self.separator_slope_input.editingFinished.emit()
                    

                for row in csv_reader:
                    
                    voltages_1.append(float(row[0]))
                    frequencies_1.append(float(row[1]))
                    voltages_2.append(float(row[2]))
                    frequencies_2.append(float(row[3]))

                self.voltages[self.selected_file_labels] = [np.array(voltages_1), np.array(voltages_2)]
                self.frequencies[self.selected_file_labels] = [np.array(frequencies_1), np.array(frequencies_2)]

        except Exception as e:
            print(f"Error loading data from {self.data_filepath}: {e}")

        self.peaks_fit.emit(np.array(self.voltages[self.selected_file_labels]), np.array(self.frequencies[self.selected_file_labels]))

        self.plot_data()

    def selected_file_changed_handler(self, labels, filename):
        self.selected_file_labels = labels

        if labels in self.voltages:
            self.peaks_fit.emit(self.voltages[labels], self.frequencies[labels])
            self.plot_data()
        else:

            self.peaks_fit.emit(np.array([]), np.array([]))


            voltage_data = self.load_files_widget.voltage_data_all[labels]
            if labels in self.middle_index_dict:
                self.middle_index_input.setText(str(self.middle_index[labels]))
            else:
                self.middle_index_input.setText(str(int(len(voltage_data) / 2)))

            if labels in self.separator_slope_dict:
                self.separator_slope_input.setText(str(self.separator_slope[labels]))
            else:
                self.separator_slope_input.setText('0')

            self.separator_slope_input.editingFinished.emit()

            # check if the file exists and load the data
            filepath = self.get_data_filepath()
            if os.path.exists(filepath):
                self.load_data()
            else:
                self.figure.clear()
                self.canvas.draw()

    def get_data_filepath(self):
        if self.selected_file_labels:
            filepath = f"{self.data_filepath}_{self.selected_file_labels}.csv"
        else:
            filepath = f"{self.data_filepath}.csv"
        return filepath
        
    def get_voltages(self):
        return self.voltages[self.selected_file_labels]
    
    def get_frequencies(self):
        return self.frequencies[self.selected_file_labels]
    
    def get_frequency_errors(self):
        return self.frequency_errors[self.selected_file_labels]

# Test the widget with sample data
def main():
    app = QApplication(sys.argv)

    voltage_data_all = [np.linspace(-5, 5, 100) for _ in range(3)]  # Sample voltage data
    frequency_data_all = [np.linspace(4, 10, 100) for _ in range(3)]  # Sample frequency data
    transmission_data_all = [np.random.rand(100, 100) for _ in range(3)]  # Random transmission data

    # Create a test dataset and save it as a pickle (this would normally be done elsewhere in your app)
    voltages = np.linspace(-5, 5, 100)  # Sample voltage data
    frequencies = np.linspace(4, 10, 100)  # Sample frequency data

    filepath = "voltage_frequency_data.csv"  # Define the CSV file path

    # Write the data to a CSV file
    with open(filepath, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        # Write headers
        csv_writer.writerow(['Voltage', 'Frequency'])

        # Write data rows
        for i in range(len(voltages)):
            row = [voltages[i], frequencies[i]]
            csv_writer.writerow(row)

    # Create the PeakFitWidget with the path to the data file
    widget = PeakFitWidget(voltage_data_all, frequency_data_all, transmission_data_all, data_filepath=filepath)
    widget.resize(800, 600)
    widget.setWindowTitle('Peak Fit Widget')
    widget.show()

    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
