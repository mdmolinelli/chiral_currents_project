# manual_filter_data_widget.py

import csv
import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget,
                             QHBoxLayout, QApplication, QSpinBox)
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector


class ManualFilterDataWidget(QWidget):
    def __init__(self, qubit, x_getter, y_getter, data_directory):
        super().__init__()

        self.qubit = qubit  # Qubit name

        self.x_getter = x_getter  # Function to get the x data
        self.y_getter = y_getter  # Function to get the y data

        self.ignored_indices_filepath = os.path.join(data_directory, f'{qubit}_ignored_indices.txt')  # Filepath to save ignored indices
        self.filtered_data_filepath = os.path.join(data_directory, f'{qubit}_filtered.csv')  # Filepath to save filtered data

        self.selected_indices = set()  # Set to store indices of selected points
        self.ignored_indices = set()  # Set to store indices that are marked to 'ignore'
        
        self.selected_color = 'red'  # Color for ignored (added) points
        self.default_color = 'blue'  # Color for default points
        self.selection_box_color = 'black'  # Color for the dotted box around selected points

        self.filtered_x_data = None
        self.filtered_y_data = None

        self.initUI()

    def initUI(self):
        self.setFixedSize(1200, 800)  # Set the desired width and height


        # Main layout
        main_layout = QVBoxLayout(self)

        # Create matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        main_layout.addWidget(self.canvas)

        # Add point and remove point buttons
        button_layout = QHBoxLayout()
        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.add_selected_point)
        button_layout.addWidget(self.add_point_btn)

        self.remove_point_btn = QPushButton("Remove Point")
        self.remove_point_btn.clicked.connect(self.remove_selected_point)
        button_layout.addWidget(self.remove_point_btn)

        main_layout.addLayout(button_layout)

        

        display_layout = QHBoxLayout()


        # List to display selected indices
        self.selected_points_widget = QListWidget(self)

        selected_points_display_layout = QVBoxLayout()

        selected_points_display_layout.addWidget(QLabel("Selected Points:"))
        selected_points_display_layout.addWidget(self.selected_points_widget)

        display_layout.addLayout(selected_points_display_layout)

        # Input two integers and add all the points in the range to the ignored list
        add_range_display_layout = QVBoxLayout()
        

        self.add_range_widget = QWidget(self)
        add_range_layout = QHBoxLayout()
        add_range_layout.addWidget(QLabel("Start:"))
        self.start_input = QSpinBox()
        add_range_layout.addWidget(self.start_input)
        add_range_layout.addWidget(QLabel("End:"))
        self.end_input = QSpinBox()
        add_range_layout.addWidget(self.end_input)
        self.add_range_widget.setLayout(add_range_layout)


        add_range_display_layout.addWidget(QLabel("Add Range:"))
        add_range_display_layout.addWidget(self.add_range_widget)

        add_range_btn = QPushButton("Add Range")
        add_range_display_layout.addWidget(add_range_btn)
        add_range_btn.clicked.connect(self.add_range)

        remove_range_btn = QPushButton("Remove Range")
        add_range_display_layout.addWidget(remove_range_btn)
        remove_range_btn.clicked.connect(self.remove_range)

        display_layout.addLayout(add_range_display_layout)

        main_layout.addLayout(display_layout)

        save_points_btn = QPushButton("Save Ignored Points")
        save_points_btn.clicked.connect(self.save_ignore_indices)
        save_points_btn.clicked.connect(self.save_filtered_data)
        main_layout.addWidget(save_points_btn)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def run_startup(self):

        # If file exists, load
        if self.ignored_indices_filepath and os.path.exists(self.ignored_indices_filepath):
            # print(f"Loading data from {self.data_filepath}")
            self.load_ignore_indices()

        # if data file exists, load filtered data
        if self.filtered_data_filepath and os.path.exists(self.filtered_data_filepath):
            self.load_filtered_data()

        self.plot_data()  # Initial plot


    def plot_data(self):
        """Plot the data points on the canvas with different colors and indicators."""
        self.figure.clear()  # Clear previous plots
        ax = self.figure.add_subplot(111)

        x_data = self.get_x_data()
        y_data = self.get_y_data()
        for i in range(len(x_data)):

            if i in self.ignored_indices:
                # Points in ignored_indices are marked red
                ax.scatter(x_data[i], y_data[i], color=self.selected_color)
            else:
                # Default point color is blue
                ax.scatter(x_data[i], y_data[i], color=self.default_color)

            # If the point is selected but not yet added, draw a black dotted box around it
            if i in self.get_selected_indices():
                ax.scatter(x_data[i], y_data[i], facecolors='none', edgecolors=self.selection_box_color, linewidth=2, s=100, linestyle='--')

        ax.set_title('Click on points to select them')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_ylim(np.min(self.y_data) - 1, np.max(self.y_data) + 1)
        self.canvas.draw()

    def on_click(self, event):
        """Handle mouse click events to select data points."""
        if event.inaxes is not None:  # Check if the click is within the plot axes
            x_click = event.xdata
            y_click = event.ydata

            x_data = self.get_x_data()
            y_data = self.get_y_data()
            # Find the closest point by computing the Euclidean distance
            distances = np.sqrt((x_data - x_click) ** 2 + (y_data - y_click) ** 2)
            closest_index = np.argmin(distances)  # Get the index of the closest point

            # Select/deselect the point with the closest index
            if closest_index in self.get_selected_indices():
                self.remove_from_selected_indices(closest_index)  # Deselect if already selected
            else:
                self.add_to_selected_indices(closest_index)  # Select the point

            self.plot_data()  # Replot to update colors

    def add_selected_point(self):
        """Add the selected point(s) to the ignored list."""
        selected_indices = self.get_selected_indices()
        if selected_indices:
            for index in selected_indices:
                if index not in self.ignored_indices:  # Prevent duplicates
                    self.ignored_indices.add(index)
                    self.selected_points_widget.addItem(str(index))  # Add to the QListWidget
            self.clear_selected_indices()  # Clear the selection after adding
            self.plot_data()  # Replot to update colors


    def remove_selected_point(self):
        """Remove the selected point from the ignored list."""


        for index in self.get_selected_indices():
            if index in self.ignored_indices:
                self.ignored_indices.remove(index)
                items = self.selected_points_widget.findItems(str(index), Qt.MatchExactly)

                # Remove from ignored_indices and the QListWidget
                for item in items:
                    row = self.selected_points_widget.row(item)
                    self.selected_points_widget.takeItem(row)

        self.clear_selected_indices()  # Clear the selection after removing
        self.plot_data()  # Replot to update colors

    def add_range(self):
        """Add all the points in the range to the ignored list."""
        start = self.start_input.value()
        end = self.end_input.value()

        for index in range(start, end + 1):
            if index not in self.ignored_indices:  # Prevent duplicates
                self.ignored_indices.add(index)
                self.selected_points_widget.addItem(str(index))

        self.plot_data()

    def remove_range(self):
        """Remove all the points in the range from the ignored list."""
        start = self.start_input.value()
        end = self.end_input.value()

        for index in range(start, end + 1):
            if index in self.ignored_indices:
                self.ignored_indices.remove(index)
                items = self.selected_points_widget.findItems(str(index), Qt.MatchExactly)

                # Remove from ignored_indices and the QListWidget
                for item in items:
                    row = self.selected_points_widget.row(item)
                    self.selected_points_widget.takeItem(row)

        self.clear_selected_indices()  # Clear the selection after removing
        self.plot_data()

    def update_data(self, x_data, y_data):
        """Update the data and replot."""
        self.x_data = x_data
        self.y_data = y_data
        self.plot_data()  # Replot with the new data

    def save_filtered_data(self):
        """Save the filtered data to a file."""
        print(f'trying to save filtered data to {self.filtered_data_filepath}')
        if self.filtered_data_filepath:
            with open(self.filtered_data_filepath, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['X', 'Y'])
                for i in range(len(self.x_data)):
                    if i not in self.ignored_indices:
                        csv_writer.writerow([self.x_data[i], self.y_data[i]])
            print(f"Filtered data saved to {self.filtered_data_filepath}")

    def load_filtered_data(self):
        """Load the filtered data from a file."""
        self.filtered_x_data = []
        self.filtered_y_data = []
        if self.filtered_data_filepath:
            with open(self.filtered_data_filepath, 'r') as file:
                csv_reader = csv.reader(file)

                counter = 0
                for row in csv_reader:
                    if counter > 0:
                        x, y = row
                        self.filtered_x_data.append(float(x))
                        self.filtered_y_data.append(float(y))
                    counter += 1
                print(f"Filtered data loaded from {self.filtered_data_filepath}")

    def save_ignore_indices(self):
        """Save the ignored indices to a file."""
        if self.ignored_indices_filepath:
            with open(self.ignored_indices_filepath, 'w') as file:
                for index in self.ignored_indices:
                    file.write(f"{index}\n")
            print(f"Ignored indices saved to {self.ignored_indices_filepath}")

    def load_ignore_indices(self):
        """Load the ignored indices from a file."""
        if self.ignored_indices_filepath:
            with open(self.ignored_indices_filepath, 'r') as file:
                for line in file:
                    index = int(line.strip())
                    self.ignored_indices.add(index)
                    self.selected_points_widget.addItem(str(index))  # Add to the QListWidget

                print(f"Ignored indices loaded from {self.ignored_indices_filepath}")

    def get_selected_indices(self):
        """Return the selected indices."""
        return self.selected_indices
    
    def clear_selected_indices(self):
        """Clear the selected indices."""
        self.selected_indices.clear()

    def add_to_selected_indices(self, index):
        """Add an index to the selected indices."""
        self.selected_indices.add(index)

    def remove_from_selected_indices(self, index):
        """Remove an index from the selected indices."""
        self.selected_indices.remove(index)

    def get_x_data(self):
        """Return the original x data."""
        return self.x_getter()
    
    def get_y_data(self):
        """Return the original y data."""
        return self.y_getter()
        
    def get_filtered_x_data(self):
        """Return the filtered voltages."""
        if not self.filtered_x_data is None:
            return self.filtered_x_data
        elif os.path.exists(self.filtered_data_filepath):
            self.load_filtered_data()
            return self.filtered_x_data
        else:
            return np.delete(np.copy(self.get_x_data()), list(self.ignored_indices))
    
    def get_filtered_y_data(self):
        """Return the filtered frequencies."""
        if not self.filtered_y_data is None:
            return self.filtered_y_data
        elif os.path.exists(self.filtered_data_filepath):
            self.load_filtered_data()
            return self.filtered_y_data
        else:
            return np.delete(np.copy(self.get_y_data()), list(self.ignored_indices))



class ManualFilterDataWidgetAvoidedCrossings(ManualFilterDataWidget):

    def __init__(self, qubit, x_getter, y_getter, data_directory):
        super().__init__(qubit, x_getter, y_getter, data_directory)

# Test the widget with sample data
def main():
    app = QApplication(sys.argv)

    # Create a test dataset
    test_x_data = np.random.rand(10) * 10  # Random values for demonstration
    test_y_data = np.random.rand(10) * 10  # Random values for demonstration

    widget = ManualFilterDataWidget(test_x_data, test_y_data)
    widget.resize(800, 600)
    widget.setWindowTitle('Manual Filter Data Widget')
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
