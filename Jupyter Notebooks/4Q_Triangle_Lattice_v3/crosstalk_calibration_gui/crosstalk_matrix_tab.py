import csv
import os

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QTextEdit
from PyQt5.QtCore import pyqtSignal

class CrosstalkMatrixTab(QWidget):

    loaded_crosstalk_matrix = pyqtSignal(str, np.ndarray)
    loaded_crosstalk_inverse_matrix = pyqtSignal(np.ndarray)
    loaded_crosstalk_offset_vector = pyqtSignal(np.ndarray)

    crosstalk_matrix_label = 'matrix'
    crosstalk_inverse_matrix_label = 'inverse'
    crosstalk_offset_vector_label = 'offset'

    def __init__(self):
        super().__init__()

        crosstalk_matrix_directory = r'V:\QSimMeasurements\Measurements\4Q_Triangle_Lattice\crosstalk_matrices'
        self.matrix_to_filename = {'matrix': os.path.join(crosstalk_matrix_directory, 'crosstalk_matrix_4.csv'),
                                   'inverse': os.path.join(crosstalk_matrix_directory, 'crosstalk_inverse_matrix_4.csv'),
                                   'offset': os.path.join(crosstalk_matrix_directory, 'crosstalk_offset_vector_4.csv')}


        self.matrix_to_filename_input = {}
        self.matrix_to_text_output = {}
        self.matrix_to_load_button = {}
        self.matrix_to_plot_output = {}



        self.matrix_labels = [self.crosstalk_matrix_label, self.crosstalk_inverse_matrix_label, self.crosstalk_offset_vector_label]

        self.crosstalk_matrix = None
        self.crosstalk_inverse_matrix = None
        self.crosstalk_offset_vector = None

        self.initUI()



    def initUI(self):
        layout = QVBoxLayout()

        for matrix_label in self.matrix_labels:
            layout.addLayout(self.createCrosstalkMatrixRow(matrix_label))

        # layout.addLayout(self.createCrosstalkMatrixRow('matrix'))
        # layout.addLayout(self.createCrosstalkMatrixRow('inverse'))
        # layout.addLayout(self.createCrosstalkMatrixRow('offset'))
        self.setLayout(layout)

    def createCrosstalkMatrixRow(self, matrix_label):
        layout = QGridLayout()

        if matrix_label == self.crosstalk_matrix_label:
            label_text = 'Crosstalk Matrix'
        elif matrix_label == self.crosstalk_inverse_matrix_label:
            label_text = 'Inverse Crosstalk Matrix'
        elif matrix_label == self.crosstalk_offset_vector_label:
            label_text = 'Offset Vector'
        else:
            print(matrix_label)

        label = QLabel(label_text)
        filename_input = QLineEdit()
        filename_input.setText(self.matrix_to_filename[matrix_label])
        self.matrix_to_filename_input[matrix_label] = filename_input

        load_button = QPushButton("Load")
        load_button.clicked.connect(lambda: self.load(matrix_label))
        self.matrix_to_load_button[matrix_label] = load_button

        matrix_output = QTextEdit()
        matrix_output.setReadOnly(True)
        self.matrix_to_text_output[matrix_label] = matrix_output

        plot_output = self.createPlotWidget()
        self.matrix_to_plot_output[matrix_label] = plot_output

        layout.addWidget(label, 0, 0)
        layout.addWidget(filename_input, 0, 1)
        layout.addWidget(load_button, 0, 2)
        layout.addWidget(matrix_output, 0, 3)
        layout.addWidget(plot_output, 0, 4)

        return layout

    def load(self, matrix_label):
        filename = self.matrix_to_filename_input[matrix_label].text()

        matrix = []

        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if len(row) > 1:
                    row_list = []
                    for value in row:
                        row_list.append(float(value))
                    matrix.append(row_list)
                else:
                    matrix.append(float(row[0]))

        matrix = np.array(matrix)

        # emit loaded crosstalk matrix
        self.loaded_crosstalk_matrix.emit(matrix_label, matrix)

        if matrix_label == self.crosstalk_matrix_label:
            self.crosstalk_matrix = matrix
            # self.loaded_crosstalk_matrix.emit(matrix)
        elif matrix_label == self.crosstalk_inverse_matrix_label:
            self.crosstalk_inverse_matrix = matrix
            # self.loaded_crosstalk_inverse_matrix.emit(matrix)
        elif matrix_label == self.crosstalk_offset_vector_label:
            # self.loaded_crosstalk_offset_vector.emit(matrix)
            self.crosstalk_offset_vector = matrix

        # Display the matrix in the text box
        self.matrix_to_text_output[matrix_label].setPlainText(np.array_str(np.round(matrix, 2)))

        # Update the plot
        self.updatePlot(matrix_label, matrix)

    def load_all(self):
        for matrix_label in self.matrix_labels:
            self.load(matrix_label)
    def createPlotWidget(self):
        figure, ax = plt.subplots()
        canvas = FigureCanvasQTAgg(figure)
        return canvas

    def updatePlot(self, matrix_label, matrix):
        plot_widget = self.matrix_to_plot_output[matrix_label]
        ax = plot_widget.figure.axes[0]
        ax.clear()

        labels = ['Q2', 'Q3', 'Q4', 'C12', 'C13', 'C23', 'C24', 'C34']

        fontsize = 6

        if matrix_label == self.crosstalk_offset_vector_label:
            ax.plot(matrix, linestyle='', marker='o')
            ax.set_xlabel('Index')
            ax.set_ylabel('Offset')
            ax.set_xticks(range(len(labels)), labels=labels, fontsize=fontsize)
        else:
            im = ax.imshow(matrix, cmap='seismic', aspect='equal', interpolation='none')
            if matrix_label == 'matrix':
                ax.set_xlabel('Flux line')
                ax.set_ylabel('Loop')
            elif matrix_label == 'inverse':
                ax.set_xlabel('Loop')
                ax.set_ylabel('Flux line')
            ax.set_xticks(range(len(labels)), labels=labels, fontsize=fontsize)
            ax.set_yticks(range(len(labels)), labels=labels, fontsize=fontsize)

            # Add colorbar
            plot_widget.figure.colorbar(im, ax=ax)

        plot_widget.draw()
