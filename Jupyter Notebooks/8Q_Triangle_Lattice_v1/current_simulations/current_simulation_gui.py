import sys
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QPushButton, QLineEdit, QLabel, QComboBox, QScrollArea, QSizePolicy, QLayout
)
from simulation_visualizer_widget import SimulationVisualizerWidget
from phase_diagram_tab import PhaseDiagramTab
import re

from current_simulation import convert_fock_to_reduced_state, generate_basis
from quantum_states import QuantumState, FockBasisState

from visualization_tab import VisualizationTab

class CurrentSimulationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Current Simulation GUI")
        self.setGeometry(100, 100, 800, 600)

        # Create the main layout
        main_layout = QVBoxLayout(self)

        # Create the tab widget
        self.tabs = QTabWidget(self)
        main_layout.addWidget(self.tabs)

        # Add the Visualization tab
        self.visualization_tab = VisualizationTab(self)
        self.tabs.addTab(self.visualization_tab, "Visualization")

        self.phase_diagram_tab = PhaseDiagramTab(self)
        self.tabs.addTab(self.phase_diagram_tab, "Phase Diagram")

        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CurrentSimulationGUI()
    gui.show()
    sys.exit(app.exec_())