import sys
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QLabel, 
    QHBoxLayout, QGraphicsView, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsTextItem
)

from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPen, QBrush

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph.graphicsItems.ColorBarItem import ColorBarItem

import qutip as qt

import matplotlib.pyplot as plt

from current_simulation import CurrentSimulation, generate_basis
import itertools


class SimulationVisualizerWidget(QWidget):
    # TODO: allow variable saturation value (colobar limits at 0-0.5 eg)# Dont
    # TODO: figure out global phase, check with 2 qubits to start# Done
    # TODO: when adding detunings to diagonal, need to properly calculate energy (use single particle hamiltonian)
    def __init__(self, num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U, periodic=False, times=None, detuning=None, phase_style='none', psi0=None):
        """
        Initialize the simulation visualizer.
        :param num_qubits: Number of qubits in the system.
        :param num_particles: Number of particles in the system.
        :param J_parallel: Coupling strength for parallel connections.
        :param J_perp: Coupling strength for perpendicular connections.
        :param phase: phase of flux through each plaqette, either provided as a list of floats or a single float, in units of pi
        :param U: Interaction strength between particles.
        :param detuning: Detuning values for each qubit.
        :periodic: Whether the lattice has periodic boundary conditions or not.
        :param times: Array of time steps for the simulation.
        :param phase_style: Style of the phase visualization ('none', 'one' or 'two'). Number of circles to use per qubit.
                            'None' means don't represent phase
        :param psi0: Initial state of the system. If None, the ground state is used.
        """
        super().__init__()

        pg.setConfigOption('background', 'w')

        # Simulation parameters
        self.num_levels = num_levels
        self.num_qubits = num_qubits
        self.num_particles = num_particles
        self.J_parallel = J_parallel
        self.J_perp = J_perp
        self.phase = phase
        self.U = U
        self.periodic = periodic
        self.detuning = detuning
        self.psi0 = psi0
        

        if times is None:
            times = np.linspace(0, 0.1, 2)
        self.times = times
        self.num_steps = len(times)

        self.phase_style = phase_style

        self.ground_state = None
        self.result = None
        self.state_vectors = None
        self.basis = None

        self.populations = None

        # Initialize simulation
        self.simulation = CurrentSimulation(
            num_levels, num_qubits, num_particles, J_parallel, J_perp, phase, U, detuning, periodic=periodic
        )



        # GUI setup
        self.mag_color_map = pg.colormap.get('Greys', source='matplotlib')
        self.phase_color_map = pg.colormap.get('hsv', source='matplotlib')
        self.correlations_color_map = pg.colormap.get('seismic', source='matplotlib')

        self.init_ui()

        # Animation control
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.current_time_idx = 0
        self.is_animating = False

        self.min_population = None
        self.max_population = None

        # draw ground state
        self.update_plot_with_state(self.get_ground_state())


    def init_ui(self):
       
        main_layout = QVBoxLayout(self)

        # Top widget: GraphicsView for the triangle lattice
        self.graphics_view = QGraphicsView()
        self.graphics_view.setMinimumSize(400, 400)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        main_layout.addWidget(self.graphics_view)

        # Bottom widget: GraphicsLayoutWidget for data plots
        self.graphics_layout = GraphicsLayoutWidget()
        main_layout.addWidget(self.graphics_layout)


        # Data plot widget
        self.data_plot_widget = self.graphics_layout.addPlot()
        # self.data_plot_widget.setBackground('w')
        self.data_plot_widget.setTitle("Data Plot")
        self.data_plot_widget.setLabel('left', "Y-Axis")
        self.data_plot_widget.setLabel('bottom', "X-Axis")
        self.data_plot_widget.addLegend()

        # Initialize the triangle lattice in the GraphicsView
        self.init_triangle_lattice()

        ### Animations control

        # Title label
        self.title_label = QLabel("Time step: 0")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_steps - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        main_layout.addWidget(self.slider)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_animation)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_animation)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)

    def init_triangle_lattice(self):
        # Qubit circles
        self.mag_circles = {}
        self.phase_circles = {}

        triangle_edge_length = 100
        circle_size = 0.5 * triangle_edge_length
        circle_radius = circle_size / 2

        scene_center_x = 0
        scene_center_y = 0

        for i in range(self.num_qubits):
            x = i / 2 * triangle_edge_length + scene_center_x
            if i % 2 == 0:
                y = scene_center_y
            else:
                y = np.sqrt(3) / 2 * triangle_edge_length + scene_center_y
            qubit = i + 1

            print(f'creating circle at {x, y}')

            # Add magnitude circle
            mag_circle = QGraphicsEllipseItem(
                x - circle_radius, y - circle_radius, circle_size, circle_size
            )
            mag_circle.setBrush(pg.mkBrush(255, 255, 255))
            mag_circle.setPen(pg.mkPen('black', width=3))
            self.scene.addItem(mag_circle)

            self.mag_circles[qubit] = mag_circle

            # Add phase circle if phase_style is 'two'
            if self.phase_style == 'two':
                phase_circle = pg.QtWidgets.QGraphicsEllipseItem(
                    x - circle_radius / 2, y - circle_radius / 2, circle_size / 2, circle_size / 2
                )
                phase_circle.setBrush(pg.mkBrush(255, 255, 255))
                phase_circle.setPen(pg.mkPen('black', width=3))
                self.scene.addItem(phase_circle)
                self.phase_circles[qubit] = phase_circle

        # Add rectangles (edges) between circles
        self.edge_markers = {}
        self.edge_labels = {}
        for i in range(self.num_qubits):
            x1 = i / 2 * triangle_edge_length + scene_center_x

            if i % 2 == 0:
                y1 = scene_center_y
            else:
                y1 = np.sqrt(3) / 2 * triangle_edge_length + scene_center_y

            # Connect to the next vertex on the same row
            if i + 2 < self.num_qubits:
                x2 = (i + 2) / 2 * triangle_edge_length + scene_center_x
                y2 = y1
                rect = self.create_rectangle(x1, y1, x2, y2, circle_radius)
                self.edge_markers[(i + 1, i + 3)] = rect

                # Add a text label above the edge
                label_x = (x1 + x2) / 2 - triangle_edge_length * 0.22
                label_y = (y1 + y2) / 2 - triangle_edge_length * 0.12
                if i % 2 == 0:
                    label_y -= triangle_edge_length * 0.15
                else:
                    label_y += triangle_edge_length * 0.15
                
                edge_label = QGraphicsTextItem('')
                edge_label.setPos(label_x, label_y)
                self.scene.addItem(edge_label)
                self.edge_labels[(i + 1, i + 3)] = edge_label

            # Connect diagonally
            if i + 1 < self.num_qubits:
                x2 = (i + 1) / 2 * triangle_edge_length + scene_center_x
                if i % 2 == 0:
                    y2 = np.sqrt(3) / 2 * triangle_edge_length + scene_center_y
                else:
                    y2 = scene_center_y
                rect = self.create_rectangle(x1, y1, x2, y2, circle_radius)
                self.edge_markers[(i + 1, i + 2)] = rect


                # Add a text label above the edge
                label_x = (x1 + x2) / 2 - triangle_edge_length * 0.38
                label_y = (y1 + y2) / 2 - triangle_edge_length * 0.1
                if i % 2 == 0:
                    label_y += triangle_edge_length * 0.1
                else:
                    label_y -= triangle_edge_length * 0.1
                
                edge_label = QGraphicsTextItem('')
                edge_label.setPos(label_x, label_y)
                self.scene.addItem(edge_label)
                self.edge_labels[(i + 1, i + 2)] = edge_label


        color_bar_width = 20
        color_bar_height = 200
        color_bar_x = triangle_edge_length * (self.num_qubits//2 + 1)
        color_bar_y = scene_center_y + np.sqrt(3)/4*triangle_edge_length - color_bar_height/2
        color_bar_x_spacing = 50
        
        self.add_color_bar(color_bar_x, color_bar_y, color_bar_width, color_bar_height, color_map=self.mag_color_map)

        if not self.phase_style == 'none':
            self.add_color_bar(color_bar_x + color_bar_x_spacing, color_bar_y, color_bar_width, color_bar_height, color_map=self.phase_color_map, title='Phase')
        
        # TODO: dynamically control the color bar based on the current data
        self.add_color_bar(color_bar_x + 2*color_bar_x_spacing, color_bar_y, color_bar_width, color_bar_height, color_map=self.correlations_color_map, title='Currents')

    def add_color_bar(self, bar_x, bar_y, bar_width, bar_height, vmin=0, vmax=1, title=None, color_map=None):
        """
        Add a custom color bar to the graphics scene with a white-to-black color map for values between 0 and 1.
        """

        if color_map is None:
            color_map = pg.colormap.get('Greys', source='matplotlib')

        # Define the color bar dimensions
        bar_width = 20
        bar_height = 200
    
        # Create the color bar as a series of rectangles
        num_steps = 100  # Number of gradient steps
        for i in range(num_steps):
            # Calculate the color for this step
            value = 1 - i / (num_steps - 1)  # Normalize to [0, 1]
            color = pg.mkColor(color_map.map(value, mode='qcolor'))
    
            # Create a rectangle for this step
            rect = QGraphicsRectItem(bar_x, bar_y + i * (bar_height / num_steps), bar_width, bar_height / num_steps)
            rect.setBrush(color)
            rect.setPen(QPen(Qt.NoPen))            
            self.scene.addItem(rect)

        # Add a border rectangle around the entire color bar
        border_rect = QGraphicsRectItem(bar_x, bar_y, bar_width, bar_height)
        border_rect.setPen(QPen(Qt.black, 2))  # Black border with width 2
        border_rect.setBrush(QBrush(Qt.NoBrush))  # No fill
        self.scene.addItem(border_rect)
    
        # Add labels for the color bar
        min_label = QGraphicsTextItem(str(vmin))
        min_label.setPos(bar_x + bar_width + 5, bar_y + bar_height - 10)
        self.scene.addItem(min_label)
    
        max_label = QGraphicsTextItem(str(vmax))
        max_label.setPos(bar_x + bar_width + 5, bar_y - 10)
        self.scene.addItem(max_label)
    
        # Add a title for the color bar
        title_item = None
        if title is None:
            title_item = QGraphicsTextItem("Population")
        else:
            title_item = QGraphicsTextItem(title)
        title_item.setPos(bar_x, bar_y - 30)
        self.scene.addItem(title_item)


    def create_arrow(self, x1, y1, x2, y2, circle_radius):
        """
        Create an arrow between two points (x1, y1) and (x2, y2).
        """
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)  # Calculate the length of the arrow
        theta = np.arctan2(dy, dx)  # Calculate the angle of the arrow

        pos_x = x1
        pos_y = y1

        # Shift pos_x and pos_y by circle_radius in the direction of theta to avoid overlap with circles
        pos_x += circle_radius * np.cos(theta)
        pos_y += circle_radius * np.sin(theta)

        length -= 2 * circle_radius  # Adjust length to account for the circles



        head_length = 0.2 * length  # Length of the arrowhead
        tail_length = length - head_length  # Length of the tail

        arrow = pg.ArrowItem(
            pos=(pos_x, pos_y),  # Adjusted position
            angle=np.degrees(theta),  # Calculate the angle of the arrow
            brush=pg.mkBrush(255, 255, 255),  # White fill
            pen=pg.mkPen('black', width=2),  # Black outline
            headLen=head_length,  # Length of the arrowhead
            headWidth=0.25*length,  # Width of the arrowhead
            tipAngle=50,  # Angle of the arrowhead
            baseAngle=0,  # Base width of the arrowhead
            tailLen=tail_length,  # Length of the tail
            tailWidth=0.2*length,  # Width of the tail
            pxMode=False  # Use absolute pixel coordinates
        )
        self.lattice_plot_widget.addItem(arrow)
        return arrow
    
    def create_rectangle(self, x1, y1, x2, y2, circle_radius):
        """
        Create a rectangle between two points (x1, y1) and (x2, y2).
        """
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)  # Calculate the length of the rectangle
        theta = np.arctan2(dy, dx)  # Calculate the angle of the rectangle

        # Adjust the starting position to account for circle radius
        pos_x = x1 + circle_radius * np.cos(theta)
        pos_y = y1 + circle_radius * np.sin(theta)

        # Adjust the length to account for the circle radius on both ends
        length -= 2 * circle_radius

        # Set the width of the rectangle
        width = 0.2 * length  # Adjust this value as needed for visual clarity

        # Create a rectangle item
        rect = pg.QtWidgets.QGraphicsRectItem(0, -width / 2, length, width)
        rect.setPen(pg.mkPen('black', width=2))  # Black outline
        rect.setBrush(pg.mkBrush(255, 255, 255))  # White fill

        # Rotate the rectangle to align with the edge
        rect.setTransformOriginPoint(0, 0)
        rect.setRotation(np.degrees(theta))

        # Move the rectangle to the correct position
        rect.setPos(pos_x, pos_y)

        # self.lattice_plot_widget.addItem(rect)
        self.scene.addItem(rect)
        return rect

    def plot_data(self, x_data, y_data_sets, labels, title="Plot", x_label="X-Axis", y_label="Y-Axis"):
        """
        Plot multiple sets of x and y data on the additional plot widget.
        :param x_data: Array of x-axis data.
        :param y_data_sets: List of arrays for y-axis data.
        :param labels: List of labels for each y-data set.
        :param title: Title of the plot.
        :param x_label: Label for the x-axis.
        :param y_label: Label for the y-axis.
        """
        self.data_plot_widget.clear()
        self.data_plot_widget.setTitle(title)
        self.data_plot_widget.setLabel('left', y_label)
        self.data_plot_widget.setLabel('bottom', x_label)

        colors = plt.cm.tab10.colors  # Use matplotlib's Tableau colormap
        for i, (y_data, label) in enumerate(zip(y_data_sets, labels)):
            color = colors[i % len(colors)]  # Cycle through colors if more data sets than colors
            pen = pg.mkPen(color=(color[0]*255, color[1]*255, color[2]*255), width=2)
            self.data_plot_widget.plot(x_data, y_data, pen=pen, name=label)

        self.data_plot_widget.setYRange(-0.1, 1.1)

    def get_ground_state(self):
        if self.ground_state is None:
            self.ground_state = self.simulation.get_resonant_ground_state()
        return self.ground_state
    
    def get_populations(self):
        if self.populations is None:
            self.populations = self.simulation.get_populations()
        return self.populations
        
    def get_simulation_result(self):
        if self.result is None:
            # psi0 = 1/np.sqrt(2)*(qt.Qobj([0, 0, 1, 0, 0, 0]) + 1j*qt.Qobj([0, 0, 0, 0, 1, 0]))
            if self.psi0 is None:
                self.psi0 = self.get_ground_state()
            self.result = self.simulation.run_simulation(self.psi0, self.times, resonant=False)
            if not self.result is None:
                self.update_plot_with_state(self.psi0)
                self.plot_data(self.times, self.get_populations(), [f'Q{i+1}' for i in range(self.num_qubits)], title='Simulation', x_label='time ($\mu$s)', y_label='Populations')
        return self.result
    
    def get_state_vectors(self):
        if self.state_vectors is None:
            result = self.get_simulation_result()
            if not result is None:
                self.state_vectors = result.states
        return self.state_vectors
    
    def get_basis(self):
        if self.basis is None:
            self.basis = generate_basis(self.num_qubits, self.num_particles, self.num_levels)
        return self.basis
    
    def get_min_population(self):
        if self.min_population is None:
            min_population = 1
            state_vectors = self.get_state_vectors()
            if not self.state_vectors is None:
                for state_vector in state_vectors:
                    contributions = self.compute_qubit_particle_number(state_vector)
                    for qubit in contributions:
                        population = contributions[qubit]
                        if population < min_population:
                            min_population = population
                self.min_population = min_population
        return self.min_population
    

    def get_max_population(self):
        if self.max_population is None:
            max_population = 0
            state_vectors = self.get_state_vectors()
            if not self.state_vectors is None:
                for state_vector in state_vectors:
                    contributions = self.compute_qubit_particle_number(state_vector)
                    for qubit in contributions:
                        population = contributions[qubit]
                        if population > max_population:
                            max_population = population
                self.max_population = max_population
        return self.max_population


    def compute_qubit_particle_number(self, state_vector):
        qubit_particle_numbers = {i+1: 0 for i in range(self.num_qubits)}
        basis = self.get_basis()
        for amp, state in zip(state_vector.data.to_array(), basis):
            amp = amp[0]
            for qubit, occ in enumerate(state):
                if occ == 1:
                    qubit_particle_numbers[qubit+1] += np.power(np.abs(amp), 2)

        return qubit_particle_numbers

    def compute_qubit_contributions(self, state_vector):
        """Compute the contributions of each qubit from the state vector."""
        contributions = {i+1: 0 + 0j for i in range(self.num_qubits)}
        basis = self.get_basis()
        for amp, state in zip(state_vector.data.to_array(), basis):
            print(f'basis_state: {state}')
            amp = amp[0]
            for i in range(len(state)):
                contributions[i+1] += amp * state[i]

        qubit_1_phase = np.angle(contributions[1])
        for qubit in contributions:
            # Normalize contributions by phase of first qubit
            contributions[qubit] *= np.exp(-1j*qubit_1_phase)
        return contributions
    
    def convert_to_one_color(self, z, cmap=plt.cm.hsv):
        """
        Maps a complex number z (|z| ≤ 1) to a color.
        - Phase (angle) → Hue
        - Magnitude → Brightness
        """
        phase = np.angle(z)  # Range [-pi, pi]
        magnitude = np.abs(z)  # Range [0,1]

        # Normalize phase to [0,1] for colormap indexing
        hue = (phase + np.pi) / (2 * np.pi)
        
        # Get base color from the colormap
        base_color = np.array(cmap(hue))  # RGBA
        
        # Adjust brightness (Scale RGB by magnitude, ignore alpha)
        brightness = 2 * magnitude
        color = brightness * base_color[:3]
        
        min_population = self.get_min_population()
        max_population = self.get_max_population()

        return pg.mkColor(np.clip(color, min_population, max_population)*255)  # Ensure valid RGB values

    def convert_phase_to_color(self, z):
        """Convert a complex number to a color based on its phase."""
        phase = np.angle(z)
        hue = (phase + np.pi) / (2 * np.pi)

        phase_color = self.phase_color_map.mapToQColor(hue)

        return phase_color
    
    def convert_population_to_color(self, population, scaled=False):
        """Convert a complex number to a color based on its phase."""

        min_population = self.get_min_population()
        max_population = self.get_max_population()

        # Linearly transform the value between 0 and 1 to min_population and max_population
        if scaled:
            if np.abs(max_population) < 0.5:
                if np.abs(max_population - min_population) < 0.001:
                    scaled_value = 0.5
                else:
                    scaled_value = (population - min_population) / (max_population - min_population)
            else:
                scaled_value = population
        else:
            scaled_value = population

        color = self.mag_color_map.mapToQColor(scaled_value)

        return color

    def update_plot_with_state(self, state_vector):
        phase_contributions = self.compute_qubit_contributions(state_vector)
        qubit_particle_numbers = self.compute_qubit_particle_number(state_vector)

        for qubit in self.mag_circles:

            phase_color = self.convert_phase_to_color(phase_contributions[qubit])

            if self.phase_style == 'one':
                mag_color = self.convert_to_one_color(qubit_particle_numbers[qubit])
            else:
                mag_color = self.convert_population_to_color(qubit_particle_numbers[qubit])

            mag_circle = self.mag_circles[qubit]
            mag_circle.setBrush(pg.mkBrush(mag_color))

            if self.phase_style == 'two':
                phase_circle = self.phase_circles[qubit]
                phase_circle.setBrush(pg.mkBrush(phase_color))

    def update_plot(self):
        """Update the plot for the current time index."""
        self.title_label.setText(f"Time step: {self.current_time_idx}")
        # print(f'current_time_idx: {self.current_time_idx}')
        state_vectors = self.get_state_vectors()
        if self.state_vectors is None or len(state_vectors) == 0:
            print("No state vectors available. Please run a simulation first.")
            return
        state_vector = state_vectors[self.current_time_idx]
        
        self.update_plot_with_state(state_vector)

        # Update slider position
        self.slider.setValue(self.current_time_idx)

        # Increment time index
        self.current_time_idx += 1
        if self.current_time_idx >= self.num_steps:
            self.current_time_idx = 0  # Loop back to the beginning

    def on_slider_change(self, value):
        """Handle slider value changes."""
        self.current_time_idx = value
        self.update_plot()

    def start_animation(self):
        """Start the animation."""
        if not self.is_animating:
            self.is_animating = True
            self.timer.start(100)  # Update every 100 ms

    def stop_animation(self):
        """Stop the animation."""
        if self.is_animating:
            self.is_animating = False
            self.timer.stop()


    def clear_simulation(self):
        self.result = None
        self.state_vectors = None
        self.basis = None 

        self.simulation = CurrentSimulation(
            self.num_levels, self.num_qubits, self.num_particles, self.J_parallel, self.J_perp, self.detuning, self.U, periodic=self.periodic
        )

        self.current_time_idx = 0
        self.is_animating = False
        self.num_steps = len(self.times)

        # reset slider
        self.slider.setMaximum(self.num_steps - 1)
        self.slider.setValue(0)

    def set_initial_state_index(self, excited_state_index):
        initial_state = self.simulation.get_excited_state(excited_state_index)

        
        print('setting initial state to excited state:')
        print(initial_state)

        for i in range(self.num_qubits):
            contribution = 0
            basis = self.get_basis()
            for amp, state in zip(initial_state.data.to_array(), basis):
                amp = amp[0]
                if state[i] == 1:
                    contribution += np.abs(amp)**2
            print(f'Qubit {i+1} population: {contribution}')

        self.set_initial_state(initial_state)

    def set_initial_state(self, initial_state):
        self.psi0 = initial_state
        self.simulation.psi0 = initial_state
        self.update_plot_with_state(initial_state)


    def plot_currents(self):

        # reset all edge colors
        for edge in self.edge_markers:
            rect = self.edge_markers[edge]
            rect.setBrush(pg.mkBrush(255, 255, 255))

        cmap = self.correlations_color_map

        currents = self.simulation.get_currents()

        all_qubits = [i+1 for i in range(self.num_qubits)]
        # generate all possible qubit pairs with j > i using itertools combinations
        qubit_pairs = list(itertools.combinations(all_qubits, 2))


        for qubit_pair in qubit_pairs:

            if qubit_pair in currents:
                print(f'Currents for qubit pair {qubit_pair}: {currents[qubit_pair]}')

                current_value = currents[qubit_pair]

                # Normalize correlation value to [0, 1] for colormap
                norm_current = (current_value + 1) / 2  # Assuming correlation values are in [-1, 1]

                # Use matplotlib's seismic colormap
                # color = cmap(norm_current)
                qcolor = self.correlations_color_map.map(norm_current, mode='qcolor')

                # Convert to QColor for PyQtGraph
                # qcolor = pg.mkColor(color[0] * 255, color[1] * 255, color[2] * 255)

                if qubit_pair in self.edge_markers:
                    rect = self.edge_markers[qubit_pair]
                    rect.setBrush(pg.mkBrush(qcolor))

                # set labels
                if qubit_pair in self.edge_labels:
                    edge_label = self.edge_labels[qubit_pair]
                    edge_label.setPlainText(f'{current_value:.2f}')

    def plot_current_correlations(self, Q_i, Q_j):

        # reset all edge colors
        for edge in self.edge_markers:
            rect = self.edge_markers[edge]
            rect.setBrush(pg.mkBrush(255, 255, 255))

        # set base pair to black
        if (Q_i, Q_j) in self.edge_markers:
            rect = self.edge_markers[(Q_i, Q_j)]
            rect.setBrush(QBrush(Qt.black))

        # set any edge that shares a vertex with the base pair to black
        for edge in self.edge_markers:
            if Q_i in edge or Q_j in edge:
                rect = self.edge_markers[edge]
                rect.setBrush(QBrush(Qt.black))


        current_correlations = self.simulation.get_current_correlations()

        all_qubits = [i+1 for i in range(self.num_qubits)]
        # generate all possible qubit pairs with j > i using itertools combinations
        qubit_pairs = list(itertools.combinations(all_qubits, 2))

        cmap = self.correlations_color_map

        # print('all correlations')
        # print(current_correlations)

        for qubit_pair in qubit_pairs:

            correlation_pair = ((Q_i, Q_j), qubit_pair)
            if not correlation_pair in current_correlations:
                correlation_pair = (qubit_pair, (Q_i, Q_j))

            
            if correlation_pair in current_correlations:
                print(f'Current correlations for qubit pair {correlation_pair}: {current_correlations[correlation_pair]}')

                correlation_value = current_correlations[correlation_pair]

                # Normalize correlation value to [0, 1] for colormap
                norm_correlation = (correlation_value + 1) / 2  # Assuming correlation values are in [-1, 1]

                # Use matplotlib's seismic colormap
                qcolor = self.correlations_color_map.map(norm_correlation, mode='qcolor')

                # Convert to QColor for PyQtGraph
                # qcolor = pg.mkColor(color[0] * 255, color[1] * 255, color[2] * 255)

                # set colors for edges not adjacent to base
                if qubit_pair in self.edge_markers:
                    if not Q_i in qubit_pair and not Q_j in qubit_pair: 
                        rect = self.edge_markers[qubit_pair]
                        rect.setBrush(pg.mkBrush(qcolor))

                # set labels for edges not adjacent to base
                if qubit_pair in self.edge_labels:
                    if not Q_i in qubit_pair and not Q_j in qubit_pair: 
                        edge_label = self.edge_labels[qubit_pair]
                        edge_label.setPlainText(f'{correlation_value:.2f}')

            elif (Q_i, Q_j) == qubit_pair:
                edge_label = self.edge_labels[qubit_pair]
                edge_label.setPlainText('base')

    def enable_simulation_buttons(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)



# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    num_qubits = 8
    num_particles = 4
    J_parallel = 1 * 2*np.pi
    J_perp = J_parallel
    detuning = [1000] * num_qubits
    detuning[0] = 0
    detuning[1] = 0
    times = np.linspace(0, 1, 101)

    """Initialize the user interface."""

    main_window = QMainWindow()
    main_window.setWindowTitle("Simulation Visualizer")
    main_window.setGeometry(0, 0, 1200, 800)

    # Main widget and layout
    visualizer_widget = SimulationVisualizerWidget(num_qubits, num_particles, J_parallel, J_perp, times, detuning,
                                      phase_style='two')
    main_window.setCentralWidget(visualizer_widget)

    main_window.show()

    sys.exit(app.exec_())



