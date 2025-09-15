import h5py
import matplotlib.pyplot as plt
import numpy as np


class Singleshot_Measurement:
    def __init__(self, filename, qubits):
        self.filename = filename
        self.qubits = qubits

        self.confusion_matrix = None

    def get_confusion_matrix(self):
        if self.confusion_matrix is None:
            self.acquire_data()
        return self.confusion_matrix
        
    def acquire_data(self):
        self.confusion_matrix = acquire_data(self.filename)

    def plot_confusion_matrix(self):
        pass


class Singleshot_1Q_Measurement(Singleshot_Measurement):

    def __init__(self, filename, qubit):
        super().__init__(filename, [qubit])
        self.qubit = qubit

        if isinstance(self.qubit, str) and self.qubit.startswith("Q"):
            try:
                self.readout_index = int(self.qubit[1:]) - 1
            except ValueError:
                self.readout_index = 0
        else:
            self.readout_index = 0

    def acquire_data(self):
        self.confusion_matrix = acquire_data(self.filename, self.readout_index)

    def plot_confusion_matrix(self):
        confusion_matrix = self.get_confusion_matrix()

        fig, ax = plt.subplots()
        cax = ax.matshow(confusion_matrix/np.sum(confusion_matrix[:,0]), cmap='viridis')
        plt.colorbar(cax, label='Percentage')
        ax.set_title(f'Confusion Matrix for Qubit {self.qubit}')
        
        labels = ['0', '1']

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel(f'Prepared State')

        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        plt.ylabel(f'Measured State')

        plt.show()

class Singleshot_2Q_Measurement(Singleshot_Measurement):
    def __init__(self, filename, qubit1, qubit2):
        super().__init__(filename, [qubit1, qubit2])
        self.qubit1 = qubit1
        self.qubit2 = qubit2

    def plot_confusion_matrix(self):
        confusion_matrix = self.get_confusion_matrix()

        fig, ax = plt.subplots()
        cax = ax.matshow(confusion_matrix/np.sum(confusion_matrix[:,0]), cmap='viridis')
        plt.colorbar(cax, label='Percentage')
        ax.set_title(f'Confusion Matrix for Qubits {self.qubit1} and {self.qubit2}')
        
        labels = ['00', '01', '10', '11']

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel(f'Prepared State')

        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        plt.ylabel(f'Measured State')

        plt.show()
    
def acquire_data(filepath, readout_index=None):
    with h5py.File(filepath, "r") as f:

        # for i in f:
            # print(f'{i}: {f}')
            # print(i)


        confusion_matrix = f['confusion_matrix'][()]

        if readout_index is not None:
            if isinstance(confusion_matrix, list):
                if len(confusion_matrix) > readout_index:
                    confusion_matrix = confusion_matrix[readout_index]
                confusion_matrix = confusion_matrix[0]
            elif isinstance(confusion_matrix, np.ndarray):
                if len(confusion_matrix.shape) == 3:
                    confusion_matrix = confusion_matrix[readout_index]
                else:
                    confusion_matrix = confusion_matrix[0]
    
    return confusion_matrix


def generate_singleshot_2Q_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\Singleshot_2Qubit\Singleshot_2Qubit_{}\Singleshot_2Qubit_{}_{}_data.h5'.format(date_code, date_code, time_code)

def generate_singleshot_1Q_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\Singleshot\Singleshot_{}\Singleshot_{}_{}_data.h5'.format(date_code, date_code, time_code)