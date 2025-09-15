import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox

from current_simulation import CurrentSimulation
from lattice_tab_widget import LatticeTabWidget

from current_simulation import generate_basis, convert_fock_to_reduced_state
from pyqtgraph import PlotWidget

class PhaseDiagramTab(LatticeTabWidget):
    def __init__(self, parent=None):

        self.lattice_parameters = None

        super().__init__(parent)

        self.psi0 = None

    def init_ui(self):

        super().init_ui()

        main_layout = self.layout()

        
        self.plot_widget = PlotWidget()
        self.plot_widget.setBackground('w')
        
        main_layout.addWidget(self.plot_widget)

    def create_control_panel(self):
        super().create_control_panel()

        # wrapper for state initialization controls
        self.state_initialization_wrapper_widget = QWidget()
        self.state_initialization_wrapper_widget.setLayout(QGridLayout())
        self.scrollable_layout.addWidget(self.state_initialization_wrapper_widget)

        # wrapper for phase diagram controls
        self.phase_diagram_wrapper_widget = QWidget()
        self.phase_diagram_wrapper_widget.setLayout(QGridLayout())
        self.scrollable_layout.addWidget(self.phase_diagram_wrapper_widget)

        self.lattice_parameter_labels = []
        for key in self.input_rows.keys():
            line_edit_widget, value_type, kwargs = self.input_rows[key]
            if not value_type == str:
                self.lattice_parameter_labels.append(key)


    def create_lattice(self):

        num_levels = self.get_input('num_levels')
        num_qubits = self.get_input('num_qubits')
        num_particles = self.get_input('num_particles')

        J_parallel = self.get_input('J_parallel')
        J_perp = self.get_input('J_perp')

        phase = self.get_input('phase')

        U = self.get_input('U')

        periodic = self.get_input('periodic')

        self.current_simulation = CurrentSimulation(num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U, periodic=periodic)

        self.create_state_initialization_panel(self.state_initialization_wrapper_widget)
        self.create_phase_diagram_control_panel(self.phase_diagram_wrapper_widget)

        self.prepare_excited_state()

        self.calculate_total_chiral_current()
        self.calculate_average_rung_current()

    
    def create_phase_diagram_control_panel(self, phase_diagram_wrapper_widget):
        phase_diagram_controls_layout = phase_diagram_wrapper_widget.layout()
        if phase_diagram_controls_layout is not None:
            phase_diagram_controls_layout.count()
            while phase_diagram_controls_layout.count():
                item = phase_diagram_controls_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.phase_diagram_wrapper_widget.setLayout(phase_diagram_controls_layout)
        else:
            phase_diagram_controls_layout = QGridLayout()
            # simulation_wrapper_widget_layout = QVBoxLayout()
            self.simulation_wrapper_widget.setLayout(phase_diagram_controls_layout)

        row = 0
        phase_diagram_controls_layout.addWidget(QLabel("Simulation Parameters"), row, 0)

        row += 1
        calculate_total_chiral_current_button = QPushButton("Calculate Total Chiral Current")
        phase_diagram_controls_layout.addWidget(calculate_total_chiral_current_button, row, 0)
        calculate_total_chiral_current_button.clicked.connect(self.calculate_total_chiral_current)

        self.total_chiral_current_label = QLabel("Total Chiral Current:")
        phase_diagram_controls_layout.addWidget(self.total_chiral_current_label, row, 1)

        row += 1
        calculate_average_rung_current_button = QPushButton("Calculate Average Rung Current")
        phase_diagram_controls_layout.addWidget(calculate_average_rung_current_button, row, 0)
        calculate_average_rung_current_button.clicked.connect(self.calculate_average_rung_current)

        self.average_rung_current_label = QLabel("Average Rung Current:")
        phase_diagram_controls_layout.addWidget(self.average_rung_current_label, row, 1)

        # sweep variables
        row += 2
        self.sweep_variable_dropdown = QComboBox()
        self.sweep_variable_dropdown.addItems(self.lattice_parameter_labels)
        self.sweep_variable_dropdown.setCurrentText('phase')
        phase_diagram_controls_layout.addWidget(QLabel("Sweep Variable:"), row, 0)
        phase_diagram_controls_layout.addWidget(self.sweep_variable_dropdown, row, 1)

        row +=1 
        self.add_input_row(phase_diagram_controls_layout, row, 'start:', 'sweep_start', default_value=0.0)
        row +=1 
        self.add_input_row(phase_diagram_controls_layout, row, 'stop:', 'sweep_stop', default_value=1.0)
        row +=1 
        self.add_input_row(phase_diagram_controls_layout, row, 'points:', 'sweep_points', default_value=11)


        # output variable
        self.outputs_labels = ['total_chiral_current', 
                               'average_rung_current', 
                               'density_imbalance',
                               'center_rung_correlations']
        
        self.output_checkboxes = []
        for label in self.outputs_labels:
            row += 1
            checkbox = QCheckBox(label)
            phase_diagram_controls_layout.addWidget(checkbox, row, 0)
            self.output_checkboxes.append(checkbox)
        



        # run sweep button
        row +=1 
        run_sweep_button = QPushButton("Run Sweep")
        phase_diagram_controls_layout.addWidget(run_sweep_button, row, 0, 1, 2)
        run_sweep_button.clicked.connect(self.run_sweep)

   
    def prepare_excited_state(self):
        # print('prepare excited state pressed with index: ', self.excited_state_index_input.text())
        # excited_state_index = self.excited_state_index_input.text()
        excited_state_index = self.get_input('excited_state_index')
        if not excited_state_index:
            excited_state_index = 0
        
        self.current_simulation.psi0 = self.current_simulation.get_resonant_excited_state(excited_state_index)

    def prepare_custom_state(self, custom_state):
        '''
        Assume custom_state is in fock basis
        '''
        reduced_basis = generate_basis(self.get_input('num_qubits'), self.get_input('num_particles'), self.get_input('num_levels'))
        reduced_state = convert_fock_to_reduced_state(reduced_basis, custom_state)

        self.current_simulation.psi0 = reduced_state


    def calculate_total_chiral_current(self):
        total_chiral_current = self.current_simulation.get_total_chiral_current()
        self.total_chiral_current_label.setText(f"Total Chiral Current: {total_chiral_current}")

    def calculate_average_rung_current(self):
        average_rung_current = self.current_simulation.get_average_rung_current()
        self.average_rung_current_label.setText(f"Average Rung Current: {average_rung_current}")

    def get_selected_output_variables(self):
        selected_outputs = []
        for i in range(len(self.outputs_labels)):
            if self.output_checkboxes[i].isChecked():
                selected_outputs.append(self.outputs_labels[i])
        return selected_outputs

    def run_sweep(self):
        # Get the sweep variable and range from the UI
        sweep_variable = self.sweep_variable_dropdown.currentText()

        sweep_start = self.get_input('sweep_start')
        sweep_stop = self.get_input('sweep_stop')
        sweep_points = self.get_input('sweep_points')

        excited_state_index = self.get_input('excited_state_index')

        sweep_array = np.linspace(sweep_start, sweep_stop, sweep_points)

        # set initial values for the simulation parameters
        for key in self.lattice_parameter_labels:
            # print(f"Setting {key} to {self.get_input(key)}")
            setattr(self.current_simulation, key, self.get_input(key))

        output_variables = self.get_selected_output_variables()

        # output_values = np.zeros((len(output_variables), sweep_points))

        output_values = []



        for i in range(sweep_points):
            # set the current value of the sweep variable
            # print(f"Setting {sweep_variable} to {sweep_array[i]}")
            setattr(self.current_simulation, sweep_variable, self.get_input(sweep_variable, str(sweep_array[i])))

            psi0 = self.current_simulation.get_resonant_excited_state(excited_state_index)

            for j in range(len(output_variables)):
                if len(output_values) <= j:
                    output_values.append([])  
                if output_variables[j] == 'total_chiral_current':
                    output_values[j].append(self.current_simulation.get_total_chiral_current(psi0))
                elif output_variables[j] == 'average_rung_current':
                    output_values[j].append(self.current_simulation.get_average_rung_current(psi0))
                elif output_variables[j] == 'density_imbalance':
                    output_values[j].append(self.current_simulation.get_density_imbalance(psi0))
                elif output_variables[j] == 'center_rung_correlations':
                    center_rung_correlations = self.current_simulation.get_center_rung_correlations(psi0)
                    for k in range(len(center_rung_correlations)):
                        if len(output_values) <= j + k:
                            output_values.append([])
                        output_values[j + k].append(center_rung_correlations[k].real)
                    # output_values[j].append(self.current_simulation.get_center_rung_correlations(psi0))

        print('output_values:')
        # output_values = np.array(output_values)

        # print(output_values)
        # print(output_values.shape)

        self.update_plot(sweep_array, output_values, labels=output_variables)

    def update_plot(self, x_datasets, y_datasets, labels=None):
        """
        Updates the plot with the given x and y datasets and optional labels.

        Parameters:
        x_datasets (np.ndarray or list of np.ndarray): x data arrays. Can be a single array or a list of arrays.
        y_datasets (list of np.ndarray): List of y data arrays.
        labels (list of str, optional): List of labels for each dataset.
        """

        # print("Updating plot with x_datasets:", x_datasets, "and y_datasets:", y_datasets)

        self.plot_widget.clear()  # Clear the existing plot

        # print(y_datasets.shape)

        # x_shape = x_datasets.shape
        # if len(x_shape) == 1:
        #     if x_shape[0] != len(y_datasets):
        #         raise ValueError("x_datasets and y_datasets must have the same length.")
        #     x_datasets = np.array([x_datasets] * len(y_datasets))
        # elif len(x_shape) == 2:
        #     if x_shape[0] != len(y_datasets):
        #         raise ValueError("x_datasets and y_datasets must have the same number of datasets.")
        # else:
        #     raise ValueError("x_datasets must be a 1D or 2D array.")
        
        cmap = plt.cm.get_cmap('tab10', len(y_datasets))  # Get the HSV colormap

        print(type(x_datasets), type(y_datasets))
        print(len(x_datasets), len(y_datasets))
        for i in range(len(y_datasets)):
            if isinstance(x_datasets[0], (list, np.ndarray)):
                # x_datasets is a list of arrays
                if isinstance(x_datasets[0][0], (list, np.ndarray)):
                    raise ValueError("x_datasets must be a 1D or 2D array.")
                if len(x_datasets) != len(y_datasets):
                    raise ValueError("x_datasets and y_datasets must have the same number of datasets.")
                x_data = x_datasets[i]
            else:
                # x_datasets is a single array to be used for all y_datasets
                print(len(x_datasets), len(y_datasets[i]))
                if len(x_datasets) != len(y_datasets[i]):
                    raise ValueError("x_datasets and y_datasets must have the same length.")
                x_data = x_datasets

            y_data = y_datasets[i]

            
            label = ''
            if len(labels) == len(y_datasets):
                label = labels[i]
            color = cmap(i)  # Get a color from the colormap
            color = tuple(int(c * 255) for c in color[:3])
            pen = {'color': color, 'width': 5}  # Set the line color and thickness (width=2)
            self.plot_widget.plot(x_data, y_data, pen=pen, name=label)  # Plot each dataset

        if labels:
            legend = self.plot_widget.addLegend()  # Add a legend if labels are provided
            legend.labelTextColor = 'k'  # Set legend text color to black
