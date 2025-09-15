import sys
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QPushButton, QLineEdit, QLabel, QComboBox, QScrollArea, QSizePolicy, QLayout
)
from simulation_visualizer_widget import SimulationVisualizerWidget
import re

from lattice_tab_widget import LatticeTabWidget
from current_simulation import generate_basis, convert_fock_to_reduced_state
from quantum_states import QuantumState, FockBasisState


class VisualizationTab(LatticeTabWidget):

    simulation_control_panel_created = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.visualizer = None


    def init_ui(self):

        super().init_ui()

        main_layout = self.layout()

        self.visualizer_wrapper_widget = QWidget()
        main_layout.addWidget(self.visualizer_wrapper_widget)
        

    def create_control_panel(self):
        super().create_control_panel()

        # wrapper for state initialization controls
        self.state_initialization_wrapper_widget = QWidget()
        self.state_initialization_wrapper_widget.setLayout(QGridLayout())
        self.scrollable_layout.addWidget(self.state_initialization_wrapper_widget)

        # wrapper for simulation controls
        self.simulation_wrapper_widget = QWidget()
        self.simulation_wrapper_widget.setLayout(QGridLayout())
        self.scrollable_layout.addWidget(self.simulation_wrapper_widget)
    
    def create_simulation_control_panel(self, simulation_wrapper_widget):

        
        # Add controls layout to the wrapper layout

        simulation_controls_layout = simulation_wrapper_widget.layout()
        if simulation_controls_layout is not None:
            simulation_controls_layout.count()
            while simulation_controls_layout.count():
                item = simulation_controls_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.simulation_wrapper_widget.setLayout(simulation_controls_layout)
        else:
            simulation_controls_layout = QGridLayout()
            # simulation_wrapper_widget_layout = QVBoxLayout()
            self.simulation_wrapper_widget.setLayout(simulation_controls_layout)

        row = 0
        simulation_controls_layout.addWidget(QLabel("Simulation Parameters"), row, 0)

        row +=1 
        self.add_input_row(simulation_controls_layout, row, 'start time:', 't_start', default_value=0.0)

        row +=1 
        default_value = 1.0
        if int(self.get_input('num_particles')) > 4:
            default_value = 0.2
        self.add_input_row(simulation_controls_layout, row, 'stop time:', 't_stop', default_value=default_value)
        
        row +=1 
        default_value = 101
        if int(self.get_input('num_qubits')) > 4:
            default_value = 21
        self.add_input_row(simulation_controls_layout, row, 'time points:', 't_points', default_value=101)


        num_qubits = self.visualizer.num_qubits

        self.custom_detuning_input = QLineEdit()
        self.custom_detuning_input.setText('1000.0')
        simulation_controls_layout.addWidget(self.custom_detuning_input, row, 2)

        self.set_qubit_detuning_buttons = {}

        def create_set_detuning_lambda(qubit):
            return lambda: self.set_qubit_detuning(qubit)

        # create set detuning buttons
        for i in range(num_qubits):
            row +=1 
            self.add_input_row(simulation_controls_layout, row, f'Q{i+1} detuning:', f'Q{i+1}_detuning', default_value=0.0, angular=True)

            set_detuning_button = QPushButton(f"Set")
            self.set_qubit_detuning_buttons[f'Q{i+1}'] = set_detuning_button
            set_detuning_button.clicked.connect(create_set_detuning_lambda(f'Q{i+1}'))
            simulation_controls_layout.addWidget(set_detuning_button, row, 2)


        row +=1 
        self.run_simulation_button = QPushButton("Run Simulation")
        self.run_simulation_button.clicked.connect(self.run_simulation)
        simulation_controls_layout.addWidget(self.run_simulation_button, row, 0, 1, 3)

       
        row +=1 
        simulation_controls_layout.addWidget(QLabel('Currents'), row, 0)

        row +=1 
        self.add_input_row(simulation_controls_layout, row, 'Q_i:', 'Q_i', default_value=1)
        
        row +=1 
        self.add_input_row(simulation_controls_layout, row, 'Q_j:', 'Q_j', default_value=2)

        row +=1 
        self.calculate_currents_button = QPushButton("Calculate Current")
        self.calculate_currents_button.clicked.connect(self.plot_currents)
        simulation_controls_layout.addWidget(self.calculate_currents_button, row, 0, 1, 3)

        row +=1 
        self.calculate_current_correlations_button = QPushButton("Calculate Current Correlations")
        self.calculate_current_correlations_button.clicked.connect(self.plot_current_correlations)
        simulation_controls_layout.addWidget(self.calculate_current_correlations_button, row, 0, 1, 3)


        
    def create_lattice(self):

        layout = self.visualizer_wrapper_widget.layout()
        if layout is not None and self.visualizer is not None:
            layout.removeWidget(self.visualizer)
        elif layout is None:
            layout = QVBoxLayout()


        num_levels = self.get_input('num_levels')
        num_qubits = self.get_input('num_qubits')
        num_particles = self.get_input('num_particles')

        J_parallel = self.get_input('J_parallel')
        J_perp = self.get_input('J_perp')

        phase = self.get_input('phase')

        detuning = self.get_input('detuning')

        U = self.get_input('U')

        peroidic = self.get_input('periodic')

        phase_style = self.get_input('phase_style')

        self.visualizer = SimulationVisualizerWidget(num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U, detuning=detuning, phase_style=phase_style, periodic=peroidic)
        self.simulation_control_panel_created.connect(self.visualizer.enable_simulation_buttons)

        layout.addWidget(self.visualizer)
        self.visualizer_wrapper_widget.setLayout(layout)

        self.create_state_initialization_panel(self.state_initialization_wrapper_widget)
        self.create_simulation_control_panel(self.simulation_wrapper_widget)

    def prepare_excited_state(self):
        # print('prepare excited state pressed with index: ', self.excited_state_index_input.text())
        # excited_state_index = self.excited_state_index_input.text()
        excited_state_index = self.get_input('excited_state_index')
        if not excited_state_index:
            excited_state_index = 0
        else:
            excited_state_index = int(excited_state_index)

        self.visualizer.set_initial_state_index(excited_state_index)

    def prepare_custom_state(self, custom_state):
        '''
        Assume custom_state is in fock basis
        '''
        reduced_basis = generate_basis(self.get_input('num_qubits'), self.get_input('num_particles'), self.get_input('num_levels'))
        reduced_state = convert_fock_to_reduced_state(reduced_basis, custom_state)

        print(reduced_basis)
        print(reduced_state)

        self.visualizer.set_initial_state(reduced_state)

    def run_simulation(self):

        print('run simulations pressed')

        layout = self.visualizer_wrapper_widget.layout()
        if layout is not None and self.visualizer is not None:
            layout.removeWidget(self.visualizer)
        elif layout is None:
            layout = QVBoxLayout()

        num_levels = self.get_input('num_levels')
        num_qubits = self.get_input('num_qubits')
        num_particles = self.get_input('num_particles')

        J_parallel = self.get_input('J_parallel')
        J_perp = self.get_input('J_perp')

        phase = self.get_input('phase')


        U = self.get_input('U')

        periodic = self.get_input('periodic')

        t_start = self.get_input('t_start')
        t_stop = self.get_input('t_stop')
        t_points = self.get_input('t_points')

        times = np.linspace(t_start, t_stop, t_points)

        detunings = np.zeros(num_qubits)
        for i in range(num_qubits):
            detunings[i] = self.get_input(f'Q{i+1}_detuning')

        phase_style = self.get_input('phase_style')

        self.visualizer = SimulationVisualizerWidget(num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U, times=times, detuning=detunings, periodic=periodic, phase_style=phase_style, psi0=self.visualizer.psi0)
        self.simulation_control_panel_created.connect(self.visualizer.enable_simulation_buttons)
        self.simulation_control_panel_created.emit()
        
        layout.addWidget(self.visualizer)
        self.visualizer_wrapper_widget.setLayout(layout)

    def plot_currents(self):
        self.visualizer.plot_currents()

    def plot_current_correlations(self):

        Q_i = self.get_input('Q_i')
        Q_j = self.get_input('Q_j')

        self.visualizer.plot_current_correlations(Q_i, Q_j)

    def __create_simulation_instance(self, num_levels, num_qubits, num_particles, J_parallel, J_perp, times=None, detuning=None):
        layout = self.visualizer_wrapper_widget.layout()
        if layout is not None and self.visualizer is not None:
            layout.removeWidget(self.visualizer)
        elif layout is None:
            layout = QVBoxLayout()        

        self.visualizer = SimulationVisualizerWidget(num_levels, num_qubits, num_particles, J_parallel, J_perp)
        self.simulation_control_panel_created.connect(self.visualizer.enable_simulation_buttons)


    def set_qubit_detuning(self, qubit):
        detuning = self.custom_detuning_input.text()
        if not detuning:
            detuning = 0
        else:
            detuning = float(detuning)

        line_edit, value_type, kwargs = self.input_rows[f'{qubit}_detuning']
        line_edit.setText(str(detuning))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CurrentSimulationGUI()
    gui.show()
    sys.exit(app.exec_())