import sys
import os
import csv

import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QTreeWidget, \
    QTreeWidgetItem, QAbstractItemView, QMessageBox, QApplication, QListWidget, QGraphicsView, QGraphicsScene, \
    QGraphicsPixmapItem, QSizePolicy, QCheckBox
from PyQt5.QtCore import Qt, QDir, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from util.data_util import get_data, plot_spec_data

class LoadFilesWidget(QWidget):

    selected_file_changed = pyqtSignal(tuple, str)

    def __init__(self, qubit, root_directory, selected_files_config_filename, measurement_type='amp', file_label_args=[]):
        '''

        Parameters
        ----------
        root_directory: directory for measurement files to search through
        selected_files_config_filename: filename that holds a list of filenames that should be added to the selected
                                        files list
        include_dict_labels: if True, label files with additional parameters in dictionary format
        '''
        super().__init__()

        self.qubit = qubit
        self.root_directory = root_directory
        self.selected_files_config_filename = selected_files_config_filename

        self.measurement_type = measurement_type
        
        if len(file_label_args) == 0:
            self.include_dict_labels = False
        else:
            self.include_dict_labels = True
        self.set_dict_labels_format(*file_label_args)

        self.voltage_data_all = None
        self.frequency_data_all = None
        self.transmission_data_all = None

        self.fit_voltages = None
        self.fit_frequencies = None
        self.middle_frequency_index = None
        self.separator_slope = 0

        self.initUI()

    def initUI(self):
        self.setFixedSize(1200, 1200)  # Set the desired width and height


        # Main Layout
        main_layout = QVBoxLayout(self)

        # Root Directory Section
        dir_layout = QHBoxLayout()
        dir_label = QLabel('Root Directory:')
        dir_label.setFont(QFont('Arial', 10))

        self.root_dir_edit = QLineEdit(self)
        self.root_dir_edit.setText(self.root_directory)
        self.root_dir_edit.textChanged.connect(self.update_root_directory)

        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.root_dir_edit)

        main_layout.addLayout(dir_layout)

        # Directory Display Section with Tree Structure
        dir_display_layout = QHBoxLayout()

        self.file_tree_widget = QTreeWidget(self)
        self.file_tree_widget.setHeaderLabel("Files and Folders")
        self.file_tree_widget.itemDoubleClicked.connect(self.item_double_clicked_handler)
        self.file_tree_widget.itemClicked.connect(self.item_selected_handler)
        self.file_tree_widget.setMouseTracking(True)  # Enable hover tracking

        self.file_tree_widget.setSelectionMode(QAbstractItemView.SingleSelection)

        # Set size policy to allow vertical expansion
        self.file_tree_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.file_tree_widget.setMinimumHeight(100)  # Optional: Set a minimum height
        self.file_tree_widget.setMinimumHeight(300)  # Optional: Set a maximum height

        if self.include_dict_labels:
            self.added_files_widget = QTreeWidget(self)
            self.added_files_widget.itemSelectionChanged.connect(self.item_selected_handler)
        else:
            self.added_files_widget = QListWidget(self)
        self.added_files_widget.setSelectionMode(QAbstractItemView.SingleSelection)

        dir_display_layout.addWidget(self.file_tree_widget)
        dir_display_layout.addWidget(self.added_files_widget)

        # Set size policy to allow vertical expansion
        self.added_files_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.added_files_widget.setMinimumHeight(300)  # Optional: Set a minimum height

        # Buttons for adding and removing files
        button_layout = QVBoxLayout()

        # Add File Button with Lambda
        add_file_layout = QHBoxLayout()
        self.add_file_btn = QPushButton("Add File", self)
        self.add_file_btn.clicked.connect(
            lambda: self.add_file(self.file_tree_widget.currentItem()))  # Pass selected item
        add_file_layout.addWidget(self.add_file_btn)

        labels_layout = QGridLayout()
        self.label_edits = []
        for i in range(len(self.labels)):
            label = self.labels[i]
            label_type = self.label_types[i]
            if label_type == bool:
                # create check box instead of QLineEdit
                label_edit = QCheckBox(self)
            else:
                label_edit = QLineEdit(self)
                label_edit.setText('0')
            self.label_edits.append(label_edit)
            labels_layout.addWidget(QLabel(label), i, 0)
            labels_layout.addWidget(label_edit, i, 1)
        add_file_layout.addLayout(labels_layout)

        button_layout.addLayout(add_file_layout)

        self.remove_file_btn = QPushButton("Remove File", self)
        self.remove_file_btn.clicked.connect(lambda: self.remove_file(self.added_files_widget.currentItem()))
        button_layout.addWidget(self.remove_file_btn)

        dir_display_layout.addLayout(button_layout)

        main_layout.addLayout(dir_display_layout)

        # save selected files button
        save_selected_files_button = QPushButton('Save Selected Files', self)
        save_selected_files_button.clicked.connect(self.save_selected_files)
        main_layout.addWidget(save_selected_files_button)

        # load files button
        load_files_button = QPushButton('Load Files', self)
        load_files_button.clicked.connect(self.load_files)
        main_layout.addWidget(load_files_button)


        # Plot Section (with a matplotlib canvas)
        
        self.plot_button = QPushButton("Plot Files", self)
        self.plot_button.clicked.connect(self.plot_files)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_button)
        plot_layout.addWidget(self.canvas)

        main_layout.addLayout(plot_layout)

        # Populate initial file tree
        self.populate_file_tree()

        # load files if self.selected_files_config_filename exists
        if os.path.exists(self.selected_files_config_filename):
            self.load_selected_files()

        # for filename in selected_files:
        #     if not self.is_file_already_added(filename):
        #         self.added_files_widget.addItem(filename)
        
    def run_startup(self):
        self.load_files()
        self.plot_files()

    def set_dict_labels_format(self, *args):
        
        labels = []
        label_types = []
        
        for label, label_type in args:
            labels.append(label)
            label_types.append(label_type)

        self.labels = labels
        self.label_types = label_types


    def update_root_directory(self, text):
        """Update the class variable root_directory whenever the QLineEdit is updated."""
        self.root_directory = text
        self.populate_file_tree()

    def populate_file_tree(self):
        """Populate the tree widget with the files and directories in the root_directory."""
        self.file_tree_widget.clear()

        if os.path.isdir(self.root_directory):
            try:
                root_item = QTreeWidgetItem(self.file_tree_widget, [self.root_directory])
                self.add_tree_items(root_item, self.root_directory)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to list files: {str(e)}")
        else:
            QMessageBox.warning(self, "Invalid Directory", "Please enter a valid directory path.")

    def add_tree_items(self, tree_item, path):
        """Recursively add subfiles and subfolders to the tree item."""
        for file_name in os.listdir(path):
            full_path = os.path.join(path, file_name)
            if os.path.isdir(full_path):
                item = QTreeWidgetItem(tree_item, [file_name])
                self.add_tree_items(item, full_path)
            else:
                if full_path.endswith('.mat'):
                    pass
                    # item = QTreeWidgetItem(tree_item, [file_name.split('.')[0]])
                    item = QTreeWidgetItem(tree_item, [file_name])

    def item_double_clicked_handler(self, item):
        child_count = item.childCount()

        if child_count == 0:
            # file
            self.add_file(item)
        else:
            # directory
            pass

    def item_selected_handler(self, *args):
        # plot image if it's a single file
        if self.include_dict_labels:
            selected_items = self.added_files_widget.selectedItems()
            if len(selected_items) > 0:
                selected_item = selected_items[0]
                if selected_item.text(0).endswith('.mat'):
                    self.plot_files()

                    item_path = self.get_item_path(selected_item)

                    self.selected_file_changed.emit(tuple(item_path[:-1]), item_path[-1])
        
    def add_file(self, item):
        """Add the selected file from file_tree_widget to the added_files_widget."""
        file_name = item.text(0)
        parent_path = self.get_item_full_path(item)

        if file_name.endswith(".mat") and os.path.isfile(parent_path):
            if not self.is_file_already_added(parent_path):
                if self.include_dict_labels:
                    # Create a QTreeWidgetItem with additional labels
                    file_item = QTreeWidgetItem([parent_path])
                    parent_item = None

                    for i in range(len(self.labels)):
                        label_text = self.label_edits[i].text()
                        parent_item = self.find_or_create_parent_item(label_text, parent_item, level=i)

                    if parent_item:
                        parent_item.addChild(file_item)
                    else:
                        self.added_files_widget.addTopLevelItem(file_item)
                else:
                    self.added_files_widget.addItem(parent_path)
        else:
            QMessageBox.warning(self, "Invalid File", "Only .mat files can be added.")

    def find_item(self, label):
        def find_item_recursive(item, label):
            if item.text(0) == label:
                return item
            else:
                for i in range(item.childCount()):
                    child = item.child(i)
                    found_item = find_item_recursive(child, label)
                    if found_item:
                        return found_item
            return None

        # recursively search through tree starting at top level items
        for i in range(self.added_files_widget.topLevelItemCount()):
            top_item = self.added_files_widget.topLevelItem(i)
            if top_item.text(0) == label:
                return top_item
            else:
                item = find_item_recursive(top_item, label)
                if item:
                    return item

    def find_or_create_parent_item(self, label_text, parent_item, level=None):
        """Find or create a parent item with the specified label text."""

        found_items = self.added_files_widget.findItems(label_text, Qt.MatchRecursive)
        if len(found_items) > 0:
            item = found_items[0]
            if level is None or self.get_item_height(item) == level:
                return item

        # If the item does not exist, create a new one
        new_item = QTreeWidgetItem([label_text])
        if parent_item:
            parent_item.addChild(new_item)
        else:
            self.added_files_widget.addTopLevelItem(new_item)
        return new_item

    def get_item_height(self, item):
        """Recursively find the height of a child in the tree, with the top level items being at height 0."""
        height = 0
        while item.parent() is not None:
            item = item.parent()
            height += 1
        return height

    def remove_file(self, item):
        """Remove the selected file from the added_files_widget."""
        if item:
            if self.include_dict_labels:
                # If using QTreeWidget, remove the item and its children
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    index = self.added_files_widget.indexOfTopLevelItem(item)
                    self.added_files_widget.takeTopLevelItem(index)
            else:
                # If using QListWidget, remove the item
                self.added_files_widget.takeItem(self.added_files_widget.row(item))

    def is_file_already_added(self, file_name):
        """Check if the file is already added to the added_files_widget."""
        if self.include_dict_labels:
            # Check for QTreeWidget
            for i in range(self.added_files_widget.topLevelItemCount()):
                if self.added_files_widget.topLevelItem(i).text(0) == file_name:
                    return True
        else:
            # Check for QListWidget
            for i in range(self.added_files_widget.count()):
                if self.added_files_widget.item(i).text() == file_name:
                    return True
        return False

    def get_item_full_path(self, item):
        """Get the full path of the selected file in the tree."""
        path = item.text(0)
        while item.parent() is not None:
            item = item.parent()
            path = os.path.join(item.text(0), path)
        return os.path.join(self.root_directory, path)

    def load_selected_files(self):
        """Load selected filenames from the specified config file."""
        try:
            with open(self.selected_files_config_filename, 'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                for row in csv_reader:

                    if len(row) == 0:
                        continue

                    if self.include_dict_labels:
                        file_item = QTreeWidgetItem([row[-1]])
                        parent_item = None

                        for i in range(len(row) - 1):
                            label_text = row[i]
                            parent_item = self.find_or_create_parent_item(label_text, parent_item)

                        if parent_item:
                            parent_item.addChild(file_item)
                        else:
                            self.added_files_widget.addTopLevelItem(file_item)

                    else:
                        filename = row[0]
                        if not self.is_file_already_added(filename):
                            self.added_files_widget.addItem(filename)

        except Exception as e:
            print(f"Error loading selected files from {self.selected_files_config_filename}: {e}")
            return []

    
    def save_selected_files(self):
        if self.include_dict_labels:
            selected_files = self.get_all_tree_widget_items(self.added_files_widget)
        else:
            selected_files = [self.added_files_widget.item(i).text() for i in range(self.added_files_widget.count())]

        # Write the selected files to a csv file
        try:
            with open(self.selected_files_config_filename, 'w') as file:

                csv_writer = csv.writer(file)

                if self.include_dict_labels:

                    for file_name in selected_files:
                        item_path = self.get_item_path(self.added_files_widget.findItems(file_name, Qt.MatchRecursive)[0])
                        # file.write(f"{file_name} {'/'.join(item_path)}\n")
                        csv_writer.writerow(item_path)

                    # selected_files = self.get_all_tree_widget_items(self.added_files_widget)
                    # for file_name in selected_files:
                    #     file.write(f"{file_name}\n")

                else:
                    selected_files = [self.added_files_widget.item(i).text() for i in range(self.added_files_widget.count())]
                    for file_name in selected_files:
                        # file.write(f"{file_name}\n")
                        csv_writer.writerow([file_name])
                
                    


            print(f"Selected files list saved to {self.selected_files_config_filename}")
        except Exception as e:
            print(f"Error saving selected files to {self.selected_files_config_filename}: {e}")

    def get_item_path(self, item):
        path = []
        while item is not None:
            path.insert(0, item.text(0))
            item = item.parent()
        return path

    def load_files(self):
        
        def add_data_to_dict(filename, data_dict, data):
            item_path = self.get_item_path(self.added_files_widget.findItems(filename, Qt.MatchRecursive)[0])
            total_data_dict = data_dict
            for label in item_path[:-1]:
                if label not in data_dict:
                    total_data_dict[label] = {}
                total_data_dict = total_data_dict[label]

            total_data_dict[item_path[-1]] = data

        self.voltage_data_all = {}
        self.frequency_data_all = {}
        self.transmission_data_all = {}

        if self.include_dict_labels:
            selected_files = self.get_all_tree_widget_items(self.added_files_widget)
            for selected_file in selected_files:

                item_path = self.get_item_path(self.added_files_widget.findItems(selected_file, Qt.MatchRecursive)[0])
                key = tuple(item_path[:-1])

                X, Y, Z = get_data(selected_file, measurement=self.measurement_type)

                self.voltage_data_all[key] = X
                self.frequency_data_all[key] = Y
                self.transmission_data_all[key] = Z
        else:
            
            # self.voltage_data_all['default'] = []
            # self.frequency_data_all['default'] = []
            # self.transmission_data_all['default'] = []

            selected_files = [self.added_files_widget.item(i).text() for i in range(self.added_files_widget.count())]

            for i in range(len(selected_files)):
                selected_file = selected_files[i]

                X, Y, Z = get_data(selected_file, measurement=self.measurement_type)

                self.voltage_data_all[i] = X
                self.frequency_data_all[i] = Y
                self.transmission_data_all[i] = Z

    def get_all_tree_widget_items(self, tree_widget):
        """Recursively get all file paths from a QTreeWidget."""
        def get_items_recursive(item):
            paths = []
            if item.childCount() == 0:
                paths.append(item.text(0))
            else:
                for i in range(item.childCount()):
                    paths.extend(get_items_recursive(item.child(i)))
            return paths

        all_paths = []
        for i in range(tree_widget.topLevelItemCount()):
            all_paths.extend(get_items_recursive(tree_widget.topLevelItem(i)))
        return all_paths
    
    def iterate_tree_widget_items(self, tree_widget, func, *args, **kwargs):
        """Recursively iterate over all QTreeWidgetItem objects in a QTreeWidget and call a function on each."""
        def iterate_items_recursive(item):
            func(item, *args, **kwargs)
            for i in range(item.childCount()):
                iterate_items_recursive(item.child(i))

        for i in range(tree_widget.topLevelItemCount()):
            iterate_items_recursive(tree_widget.topLevelItem(i))

    def update_plot_parameters(self, fit_voltages=None, fit_frequencies=None, middle_frequency_index=None, separator_slope=None):

        if fit_voltages is not None:
            self.fit_voltages = fit_voltages

        if fit_frequencies is not None:
            self.fit_frequencies = fit_frequencies

        if middle_frequency_index is not None:
            self.middle_frequency_index = middle_frequency_index

        if separator_slope is not None:
            self.separator_slope = separator_slope

        self.plot_files()

    def plot_files(self):
        """Print the selected files to be plotted."""

        if self.voltage_data_all is None or self.frequency_data_all is None or self.transmission_data_all is None:
            self.load_files()

        if len(self.voltage_data_all) > 0:

            if self.include_dict_labels:
                selected_items = self.added_files_widget.selectedItems()
                if len(selected_items) > 0:
                    selected_item = selected_items[0]
                    item_path = self.get_item_path(selected_item)

                    key = tuple(item_path[:-1])

                    voltage_data_all = [self.voltage_data_all[key]]
                    frequency_data_all = [self.frequency_data_all[key]]
                    transmission_data_all = [self.transmission_data_all[key]]

                    # voltage_dict = self.voltage_data_all
                    # frequency_dict = self.frequency_data_all
                    # transmission_dict = self.transmission_data_all

                    # try:
                    #     for label in item_path[:-1]:
                    #         voltage_dict = voltage_dict[label]
                    #         frequency_dict = frequency_dict[label]
                    #         transmission_dict = transmission_dict[label]

                    #     voltage_data_all = [voltage_dict[item_path[-1]]]
                    #     frequency_data_all = [frequency_dict[item_path[-1]]]
                    #     transmission_data_all = [transmission_dict[item_path[-1]]]
                    # except KeyError:
                    #     QMessageBox.warning(self, "Selected file not loaded yet.")
                    #     return
                else:
                    return
            else:
                voltage_data_all = [self.voltage_data_all[key] for key in self.voltage_data_all]
                frequency_data_all = [self.frequency_data_all[key] for key in self.frequency_data_all]
                transmission_data_all = [self.transmission_data_all[key] for key in self.transmission_data_all]

                # voltage_data_all = self.voltage_data_all['default']
                # frequency_data_all = self.frequency_data_all['default']
                # transmission_data_all = self.transmission_data_all['default']


            plot_spec_data(self.figure, voltage_data_all, frequency_data_all, transmission_data_all, qubit_name=self.qubit, 
                           fit_voltages=self.fit_voltages, fit_frequencies=self.fit_frequencies, middle_frequency_index=self.middle_frequency_index, separator_slope=self.separator_slope)   

            self.canvas.draw()

            

# Running the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoadFilesWidget(r'V:\QSimMeasurements\Measurements\5QV1_Triangle_Lattice', r'C:\Users\mattm\OneDrive\Desktop\Research\Projects\Triangle Lattice\Jupyter Notebooks\5Q_Triangle_Lattice\measurement_analysis_gui\test_selected_files.txt')
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
