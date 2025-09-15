import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, \
    QPushButton, QGroupBox, QScrollArea
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QTextEdit

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Measurement Control')

        tabs = QTabWidget()
        tabs.addTab(VoltageSweepTab(), "Voltage Sweep")
        tabs.addTab(QWidget(), "Spectroscopy Measurements")
        tabs.addTab(QWidget(), "Calibration")
        tabs.addTab(CrosstalkMatrixTab(), "Crosstalk Matrix")  # New tab

        self.setCentralWidget(tabs)



class RandomVoltagesTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

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

        layout.addWidget(QLabel("flux points:"), 1, 0)
        layout.addWidget(QLabel("lower bound:"), 2, 0)
        layout.addWidget(QLabel("upper bound:"), 3, 0)

        self.flux_points = []
        self.lower_bounds = []
        self.upper_bounds = []

        validator = QDoubleValidator()

        for i in range(8):
            flux_point = QLineEdit()
            flux_point.setValidator(validator)
            flux_point.setText("0")
            layout.addWidget(flux_point, 1, i+1)
            self.flux_points.append(flux_point)

            lower_bound = QLineEdit()
            lower_bound.setValidator(validator)
            lower_bound.setText("0.3")
            layout.addWidget(lower_bound, 2, i+1)
            self.lower_bounds.append(lower_bound)

            upper_bound = QLineEdit()
            upper_bound.setValidator(validator)
            upper_bound.setText("0.7")
            layout.addWidget(upper_bound, 3, i+1)
            self.upper_bounds.append(upper_bound)

        generate_button = QPushButton("Generate Random Fluxes")
        layout.addWidget(generate_button, 4, 0, 1, 9)

        # Scrollable text box
        self.generated_fluxes_display = QTextEdit()
        self.generated_fluxes_display.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.generated_fluxes_display)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, 5, 0, 1, 9)

        group.setLayout(layout)
        return group

    def createConvertVoltagesGroup(self):
        group = QGroupBox("Convert to Voltages")
        layout = QGridLayout()
        layout.addWidget(QLabel("single channel threshold:"), 0, 0)
        self.single_channel_threshold = QLineEdit()
        self.single_channel_threshold.setValidator(QDoubleValidator())
        layout.addWidget(self.single_channel_threshold, 0, 1)
        convert_button = QPushButton("Convert to Voltages")
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
        layout.addWidget(export_button)
        group.setLayout(layout)
        return group


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
        tabs.addTab(RandomVoltagesTab(), "Random Voltages")
        tabs.addTab(VoltageSweepSubTab(), "Voltage Sweep")
        layout.addWidget(tabs)
        self.setLayout(layout)


class CrosstalkMatrixTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.addLayout(self.createCrosstalkMatrixRow("Crosstalk Matrix"))
        layout.addLayout(self.createCrosstalkMatrixRow("Inverse Crosstalk Matrix"))
        layout.addLayout(self.createCrosstalkMatrixRow("Offset Vector"))
        self.setLayout(layout)

    def createCrosstalkMatrixRow(self, label_text):
        layout = QGridLayout()
        label = QLabel(label_text)
        filename_input = QLineEdit()
        load_button = QPushButton("Load")
        matrix_output = QTextEdit()
        matrix_output.setReadOnly(True)
        plot_output = self.createPlotWidget()
        layout.addWidget(label, 0, 0)
        layout.addWidget(filename_input, 0, 1)
        layout.addWidget(load_button, 0, 2)
        layout.addWidget(matrix_output, 0, 3)
        layout.addWidget(plot_output, 0, 4)
        return layout

    def createPlotWidget(self):
        figure, ax = plt.subplots()
        canvas = FigureCanvasQTAgg(figure)
        return canvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Measurement Control')
        tabs = QTabWidget()
        tabs.addTab(VoltageSweepTab(), "Voltage Sweep")
        tabs.addTab(QWidget(), "Spectroscopy Measurements")
        tabs.addTab(QWidget(), "Calibration")
        tabs.addTab(CrosstalkMatrixTab(), "Crosstalk Matrix")  # New tab
        self.setCentralWidget(tabs)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
