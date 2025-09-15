import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget

from calibration_tab import CalibrationTab
from spectroscopy_tab import SpectroscopyMeasurementsTab
from crosstalk_matrix_tab import CrosstalkMatrixTab
from voltage_sweep_tab import VoltageSweepTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('4Q Triangle Lattice calibration')

        # create tab instances
        crosstalk_matrix_tab = CrosstalkMatrixTab()
        voltage_sweep_tab = VoltageSweepTab()
        spectroscopy_tab = SpectroscopyMeasurementsTab()
        calibration_tab = CalibrationTab(spectroscopy_tab)

        # connect signals and slots
        crosstalk_matrix_tab.loaded_crosstalk_matrix.connect(voltage_sweep_tab.random_voltages_tab.set_crosstalk_matrix)

        # load crosstalk matrices
        crosstalk_matrix_tab.load_all()

        # add tabs in correct order
        tabs = QTabWidget()

        tabs.addTab(voltage_sweep_tab, "Voltage Sweep")
        tabs.addTab(spectroscopy_tab, "Spectroscopy Measurements")
        tabs.addTab(calibration_tab, "Calibration")
        tabs.addTab(crosstalk_matrix_tab, "Crosstalk Matrix")  # New tab

        self.setCentralWidget(tabs)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
    print('yes')
