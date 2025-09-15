import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QDesktopWidget

from custom_tabs.spectroscopy_tab import SpectroscopyMeasurementsTab
from custom_tabs.avoided_crossings_tab import AvoidedCrossingMeasurementsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.root_directory = os.path.dirname(os.path.abspath(__file__))
        self.measurements_directory = r'V:\QSimMeasurements\Measurements\5QV1_Triangle_Lattice'
        # self.measurements_directory = r'C:\Users\mattm\OneDrive\Desktop\Research\Measurements\5Q_Triangle_lattice\Cooldown September 13, 2024\qubit_spectroscopy'        
        
        self.initUI()


    def initUI(self):
        self.setWindowTitle('5Q Triangle Lattice calibration')

        # create tab instances
        # spectroscopy_tab = SpectroscopyMeasurementsTab(self.root_directory, self.measurements_directory)
        avoided_crossing_tab = AvoidedCrossingMeasurementsTab(self.root_directory, self.measurements_directory)

        # add tabs in correct order
        tabs = QTabWidget()

        # tabs.addTab(spectroscopy_tab, "Spectroscopy Measurements")
        tabs.addTab(avoided_crossing_tab, "Avoided Crossing Measurements")

        self.setCentralWidget(tabs)

        # Maximize the window
        self.showMaximized()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
