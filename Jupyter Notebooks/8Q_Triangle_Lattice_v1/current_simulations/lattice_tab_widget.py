import sys
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox,
    QPushButton, QLineEdit, QLabel, QComboBox, QScrollArea, QSizePolicy, QLayout
)
from simulation_visualizer_widget import SimulationVisualizerWidget
import re

from current_simulation import convert_fock_to_reduced_state, generate_basis
from quantum_states import QuantumState, FockBasisState


class LatticeTabWidget(QWidget):

    simulation_control_panel_created = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.input_rows = {}

        self.init_ui()

    def init_ui(self):

        self.resize(1200, 800)  

        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.create_control_panel()

        # Set the main layout
        self.setWindowTitle("Current Simulation GUI")
        

    def create_control_panel(self):

        main_layout = self.layout()

        # Create a scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow resizing of the scrollable content
        main_layout.addWidget(scroll_area)

        scrollable_widget = QWidget()
        self.scrollable_layout = QVBoxLayout(scrollable_widget)
        self.scrollable_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        lattice_controls_layout = QGridLayout()

        

        row = 0
        lattice_controls_layout.addWidget(QLabel("Lattice Parameters"), row, 0)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'Number of levels:', 'num_levels', default_value=2)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'Number of qubits:', 'num_qubits', default_value=4)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'Number of particles:', 'num_particles', default_value=2)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'J_parallel:', 'J_parallel', default_value=1.0, angular=True)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'J_perp:', 'J_perp', default_value=1.0, angular=True)

        row +=1 
        # if periodic, number of phase values must be num_qubits - 1, else num_qubits - 2
        num_couplers_lambda = lambda: self.get_input('num_qubits') - 1 if self.get_input('periodic') else self.get_input('num_qubits') - 2
        self.add_input_row(lattice_controls_layout, row, 'phase:', 'phase', default_value=0.0, array='optional', array_length=num_couplers_lambda, half_angular=True)

        row +=1
        num_qubits_lambda = lambda: self.get_input('num_qubits')
        self.add_input_row(lattice_controls_layout, row, 'detuning:', 'detuning', default_value=0.0, array='optional', array_length=num_qubits_lambda, angular=True)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'U:', 'U', default_value=20.0, angular=True)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'periodic BCs:', 'periodic', default_value=False)

        row +=1 
        self.add_input_row(lattice_controls_layout, row, 'Phase style', 'phase_style', default_value='none')

        # row +=1 
        # self.add_input_row(lattice_controls_layout, row, 'start time:', 't_start', default_value=0)
        # row +=1 
        # self.add_input_row(lattice_controls_layout, row, 'stop time:', 't_stop', default_value=1)
        # row +=1 
        # self.add_input_row(lattice_controls_layout, row, 'time points:', 't_points', default_value=1)

        row +=1 
        self.create_lattice_button = QPushButton("Create Lattice")
        self.create_lattice_button.clicked.connect(self.create_lattice)
        lattice_controls_layout.addWidget(self.create_lattice_button, row, 0, 1, 2)

        # row +=1 
        # self.prepare_excited_state_button = QPushButton("Prepare Excited State")
        # self.prepare_excited_state_button.clicked.connect(self.prepare_excited_state)
        # lattice_controls_layout.addWidget(self.prepare_excited_state_button, row, 0)

        # self.excited_state_index_input = QLineEdit()
        # self.excited_state_index_input.setValidator(QIntValidator(0, 100))
        # self.excited_state_index_input.setPlaceholderText("Excited State Index")
        
        # lattice_controls_layout.addWidget(self.excited_state_index_input, row, 1)

        # Add controls layout to the main layout
        self.scrollable_layout.addLayout(lattice_controls_layout)

        scroll_area.setWidget(scrollable_widget)


    def add_input_row(self, layout, row, label, dict_label=None, default_value=None, **kwargs):

        if dict_label is None:
            dict_label = label

        if default_value is None:
            default_value = 0.0
        value_type = type(default_value)

        label_widget = QLabel(label)

        line_edit_widget = None
        if value_type == bool:
            # create checkbox
            line_edit_widget = QCheckBox()
            line_edit_widget.setChecked(default_value)
        else:
            line_edit_widget = QLineEdit()
            line_edit_widget.setText(str(default_value))


        self.input_rows[dict_label] = (line_edit_widget, value_type, kwargs)

        layout.addWidget(label_widget, row, 0)
        layout.addWidget(line_edit_widget, row, 1)

    def get_input(self, dict_label, value=None):
        line_edit, value_type, kwargs = self.input_rows[dict_label]


        if value is None:       
            if value_type == bool:
                value = line_edit.isChecked()
            else:
                value = line_edit.text()
        

        if 'array' in kwargs and kwargs['array'] == 'optional':

            value = value.split(',')
            value = [value_type(v) for v in value]
        

            array_length = 1
            if 'array_length' in kwargs:
                array_length = kwargs['array_length']
                if callable(array_length):
                    array_length = array_length()

            if len(value) == 1:
                value = np.full(array_length, value[0])
            elif len(value) == array_length:
                value = np.array(value)
            else:
                raise ValueError(f"Input {dict_label} must be of length {array_length}")
        else:
            value = value_type(value)



        if 'angular' in kwargs:
            if kwargs['angular']:
                value *= 2*np.pi

        if 'half_angular' in kwargs:
            if kwargs['half_angular']:
                value *= np.pi


        return value
        
    def create_state_initialization_panel(self, state_initialization_wrapper_widget):
        
        # Add controls layout to the wrapper layout

        state_initialization_layout = state_initialization_wrapper_widget.layout()
        if state_initialization_layout is not None:
            state_initialization_layout.count()
            while state_initialization_layout.count():
                item = state_initialization_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.state_initialization_wrapper_widget.setLayout(state_initialization_layout)
        else:
            state_initialization_layout = QGridLayout()
            # state_initialization_wrapper_widget_layout = QVBoxLayout()
            self.state_initialization_wrapper_widget.setLayout(state_initialization_layout)

        row = 0
        state_initialization_layout.addWidget(QLabel("State Initialization"), row, 0)

        row +=1 
        self.add_input_row(state_initialization_layout, row, 'Excited State Index:', 'excited_state_index', default_value=0)

        row +=1 
        self.prepare_excited_state_button = QPushButton("Prepare Excited State")
        self.prepare_excited_state_button.clicked.connect(self.prepare_excited_state)
        state_initialization_layout.addWidget(self.prepare_excited_state_button, row, 0, 1, 2)

        row += 1
        self.add_input_row(state_initialization_layout, row, 'Custom State:', 'custom_state', default_value='0')

        row +=1 
        self.prepare_custom_state_button = QPushButton("Prepare Custom State")
        self.prepare_custom_state_button.clicked.connect(self.prepare_excited_state)
        state_initialization_layout.addWidget(self.prepare_custom_state_button, row, 0, 1, 2)

        custom_state_templates = []
        custom_state_labels = []

        num_qubits = self.get_input('num_qubits')
        num_particles = self.get_input('num_particles')

        if num_particles == 1:
            for i in range(num_qubits):
                basis_state = np.zeros(num_qubits)
                basis_state[i] = 1
                custom_state_templates.append(QuantumState([1], [FockBasisState(basis_state)]))
                custom_state_labels.append(f'Q{i+1}')

        if num_qubits == 4:
            if num_particles == 1:
                total_chiral_current_state = 1/np.sqrt(2)*QuantumState([1, 1j], [FockBasisState([1, 0, 0, 0]), FockBasisState([0, 0, 1, 0])])
            
                custom_state_templates.append(total_chiral_current_state)

                custom_state_labels.append('total chiral current')
            elif num_particles == 2:

                for i, basis_state in enumerate(generate_basis(num_qubits, num_particles, 2)):
                    custom_state_templates.append(QuantumState([1], [FockBasisState(basis_state)]))
                    custom_state_labels.append(f'basis state{i+1}')

                positive_current_state = 1/2*QuantumState([1, 1j, 1j, -1], [FockBasisState([1, 0, 1, 0]), FockBasisState([1, 0, 0, 1]),
                                                                            FockBasisState([0, 1, 1, 0]), FockBasisState([0, 1, 0, 1])])
                
                negative_current_state = 1/2*QuantumState([1, -1j, -1j, -1], [FockBasisState([1, 0, 1, 0]), FockBasisState([1, 0, 0, 1]),
                                                                              FockBasisState([0, 1, 1, 0]), FockBasisState([0, 1, 0, 1])])
                counter_current_state = 1/np.sqrt(2)*(positive_current_state + negative_current_state)

                custom_state_templates.append(positive_current_state)
                custom_state_templates.append(negative_current_state)
                custom_state_templates.append(counter_current_state)

                custom_state_labels.append('positive current')
                custom_state_labels.append('negative current')
                custom_state_labels.append('opposite currents')

                

                positive_current_state_2 = 1/2*QuantumState([1, 1j, 1j, -1], [FockBasisState([1, 1, 0, 0]), FockBasisState([1, 0, 0, 1]),
                                                                            FockBasisState([0, 1, 1, 0]), FockBasisState([0, 0, 1, 1])])
                
                negative_current_state_2 = 1/2*QuantumState([1, -1j, -1j, -1], [FockBasisState([1, 1, 0, 0]), FockBasisState([1, 0, 0, 1]),
                                                                            FockBasisState([0, 1, 1, 0]), FockBasisState([0, 0, 1, 1])])
                counter_current_state_2 = 1/np.sqrt(2)*(positive_current_state_2 + negative_current_state_2)

                custom_state_templates.append(positive_current_state_2)
                custom_state_templates.append(negative_current_state_2)
                custom_state_templates.append(counter_current_state_2)

                custom_state_labels.append('positive current 2')
                custom_state_labels.append('negative current 2')
                custom_state_labels.append('opposite currents 2')

                # total_chiral_current_state = 1/2*QuantumState([1, 1j, 1, 1j], [FockBasisState([1, 1, 0, 0]), FockBasisState([1, 0, 0, 1]),
                                                                            # FockBasisState([0, 1, 1, 0]), FockBasisState([0, 0, 1, 1])])

                total_chiral_current_state = 1/2*QuantumState([1, 1j, 1j, -1], [FockBasisState([0, 1, 1, 0]), FockBasisState([0, 0, 1, 1]),
                                                                            FockBasisState([1, 1, 0, 0]), FockBasisState([1, 0, 0, 1])])
            
                custom_state_templates.append(total_chiral_current_state)

                custom_state_labels.append('total chiral current')



                x_eigenstates = 1/2*QuantumState([1, 1, 1, 1], [FockBasisState([1, 1, 0, 0]), FockBasisState([1, 0, 0, 1]),
                                                                FockBasisState([0, 1, 1, 0]), FockBasisState([0, 0, 1, 1])])
                custom_state_templates.append(x_eigenstates)
                custom_state_labels.append('x eigenstates')
            
            


        def create_custom_template_lambda(custom_state):
            return lambda: self.prepare_custom_state(custom_state)

        for i in range(len(custom_state_templates)):
            row += 1
            custom_state_template_button = QPushButton(custom_state_labels[i])
            state_initialization_layout.addWidget(custom_state_template_button, row, 0, 1, 2)
            custom_state_template_button.clicked.connect(create_custom_template_lambda(custom_state_templates[i]))



    def create_lattice(self):
        pass
        