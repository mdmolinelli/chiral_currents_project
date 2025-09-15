from itertools import product

import h5py
import matplotlib.pyplot as plt
import numpy as np


class PopulationShotsBase:

    ramp=False

    def __init__(self, filename, singleshot_measurement_dict):
        self.filename = filename
        self.singleshot_measurement_dict = singleshot_measurement_dict
        
        self.confusion_matrices = None

        self.population_shots = None
        self.population_average = None
        self.population_corrected = None

        self.counts = None
        self.counts_corrected = None

        self.num_qubits = None
        self.readout_qubits = None
        self.readout_indices = None

        self.standard_deviation = None
        self.standard_deviation_corrected = None
        
        self.covariance = None
        self.covariance_corrected = None
        

    def get_population_shots(self):
        if self.population_shots is None:
            self.acquire_data()
        return self.population_shots
    
    def get_population_average(self):
        if self.population_average is None:
            self.population_average = np.mean(self.get_population_shots(), axis=-1)
        return self.population_average
    
    def get_population_corrected(self):

        confusion_matrices = self.get_confusion_matrices()

        if self.population_corrected is None:

            population = self.get_population_average()
            self.population_corrected = np.zeros_like(population)

            for i in range(self.population_corrected.shape[0]):

                population_vector_measured = np.array([1-population[i], population[i]])

                confusion_matrix_inverse = np.linalg.inv(confusion_matrices[i])

                population_vector_measured = confusion_matrix_inverse @ population_vector_measured

                self.population_corrected[i] = population_vector_measured[1]

        return self.population_corrected

    def get_confusion_matrices(self):
        if self.confusion_matrices is None:
            self.acquire_data()
        return self.confusion_matrices

    def get_readout_qubits(self):
        if self.readout_qubits is None:
            self.acquire_data()
        return self.readout_qubits
    
    def get_readout_indices(self):
        if self.readout_indices is None:
            self.readout_indices = [i - 1 for i in self.get_readout_qubits()]
        return self.readout_indices
    
    def get_num_qubits(self):
        if self.num_qubits is None:
            self.num_qubits = len(self.get_readout_qubits())
        return self.num_qubits

    def get_counts(self):
        if self.counts is None:
            self.acquire_data()
        return self.counts
    
    def get_counts_corrected(self):
        if self.counts_corrected is None:
            self.correct_counts()
        return self.counts_corrected
    
    def get_standard_deviation(self):
        if self.standard_deviation is None:

            # for binomial distribution, standard deviation is given by the formula:
            # σ = √(p(1-p))

            population_average = self.get_population_average()
            self.standard_deviation = np.sqrt(population_average * (1 - population_average))

        return self.standard_deviation
    
    def get_standard_deviation_corrected(self):
        if self.standard_deviation_corrected is None:

            # for binomial distribution, standard deviation is given by the formula:
            # σ = √(p(1-p))

            population_corrected = self.get_population_corrected()
            self.standard_deviation_corrected = np.sqrt(population_corrected * (1 - population_corrected))

        return self.standard_deviation_corrected

    def get_covariance(self):

        num_qubits = self.get_num_qubits()

        bitstrings = list(product([0, 1], repeat=num_qubits))

        if self.covariance is None:

            counts = self.get_counts()
            population_average = self.get_population_average()

            self.covariance = np.zeros([num_qubits, num_qubits] + list(population_average.shape[1:]))

            for i in range(num_qubits):
                for j in range(num_qubits):

                    if i == j:
                        self.covariance[i, i] = population_average[i] * (1 - population_average[i])
                    else:
                        bitstring = [0] * num_qubits
                        bitstring[i] = 1
                        bitstring[j] = 1

                        bitstring_index = bitstrings.index(tuple(bitstring))

                        # variance is given by the formula:
                        # σ² = <n1n2> - <n1><n2>
                        # <n1n2> is exactly the average number of occurances of 11 of qubits i and j
                        self.covariance[i, j] = counts[bitstring_index] / np.sum(counts, axis=0) - population_average[i] * population_average[j]


            # population_average = self.get_population_average()

            # counts = self.get_counts()

            # # <n1n2> is exactly the average number of occurances of 11
            # # variance is given by the formula:
            # # σ² = <n1n2> - <n1><n2>
            # self.covariance = counts[3]/np.sum(counts, axis=0) - population_average[0] * population_average[1]

        return self.covariance
    
    def get_covariance_corrected(self):
        if self.covariance_corrected is None:

            num_qubits = self.get_num_qubits()

            counts = self.get_counts()
            population_corrected = self.get_population_corrected()

            self.covariance_corrected = np.zeros([num_qubits, num_qubits] + list(population_corrected.shape[1:]))

            for i in range(num_qubits):
                for j in range(num_qubits):

                    if i == j:
                        self.covariance_corrected[i, i] = population_corrected[i] * (1 - population_corrected[i])
                    elif i > j:
                        self.covariance_corrected[i, j] = self.covariance_corrected[j, i]
                    else:

                        reduced_counts = reduce_counts(counts, i, j, num_qubits)

                        singleshot_measurement = self.singleshot_measurement_dict[(i, j)]

                        confusion_matrix_inv = np.linalg.inv(singleshot_measurement.get_confusion_matrix())

                        reduced_counts_corrected = confusion_matrix_inv @ reduced_counts

                        # variance is given by the formula:
                        # σ² = <n1n2> - <n1><n2>
                        # <n1n2> is exactly the average number of occurances of 11 of qubits i and j
                        self.covariance_corrected[i, j] = reduced_counts_corrected[3]/np.sum(reduced_counts_corrected, axis=0) - population_corrected[i] * population_corrected[j]

        return self.covariance_corrected

    def acquire_data(self):
        self.population_shots, self.counts, self.readout_qubits, self.confusion_matrices = acquire_data(self.filename, self.ramp)

        if self.counts is None:
            self.create_counts()

        if len(self.counts.shape) > 1:
            if not isinstance(self, PopulationShotsTimeSweepBase):
                raise ValueError(f'Counts has {len(self.counts.shape)} dimensions. If this is more than one, use PopulationShotsTimeSweepBase or child instead.')

    def create_counts(self):
        '''
        If counts is not provided in the file this function is used to generate the counts from
        the population shots data. This function also handles having additional axis, e.g. time.
        Ensure thats shots is the last axis.
        '''
        population_shots = self.get_population_shots()
        bit_values = 0
        num_qubits = self.get_num_qubits()

        num_qubits = population_shots.shape[0]
        num_bitstrings = 2**num_qubits
        
        # The last axis is always shots
        shots_axis = -1
        num_shots = population_shots.shape[shots_axis]
        
        # All middle dimensions (everything except first and last)
        middle_shape = population_shots.shape[1:-1]
        
    
        # Has middle dimensions: (num_qubits, ..., shots)
        total_middle_size = np.prod(middle_shape)
        counts_shape = (num_bitstrings,) + middle_shape
        self.counts = np.zeros(counts_shape, dtype=int)

        # Reshape to flatten middle dimensions
        pop_reshaped = population_shots.reshape(num_qubits, total_middle_size, num_shots)
        bit_weights = 2**(np.arange(num_qubits - 1, -1, -1))

        for i in range(total_middle_size):
            bit_values = np.dot(bit_weights, pop_reshaped[:, i, :]).astype(int)
            flat_counts = np.bincount(bit_values, minlength=num_bitstrings)
            
            # Convert flat index back to multi-dimensional index
            multi_idx = np.unravel_index(i, middle_shape)
            # Now assign to (bitstring_idx, *multi_idx)
            self.counts[(slice(None),) + multi_idx] = flat_counts
        
        
    def correct_counts(self, readout_indices=None):

        readout_indices = self.get_readout_indices()

        if len(readout_indices) > 2:
            if readout_indices is None:
                raise ValueError("readout_indices must be provided for more than 2 qubits.")
            else:
                raise RuntimeError('unimplemented for more than 2 qubits, even with readout_indices provided.')

        counts = self.get_counts()


        confusion_matrix = self.singleshot_measurement_dict[tuple(readout_indices)].get_confusion_matrix()

        inverse_confusion_matrix = np.linalg.inv(confusion_matrix)

        self.counts_corrected = inverse_confusion_matrix @ counts

    def plot_counts(self, corrected=False, both=False):

        counts = self.get_counts()
        counts_corrected = self.get_counts_corrected()

        if both:
            alpha = 0.7
        else:
            alpha = 1.0

        plt.figure(figsize=(10, 5))
        if both or not corrected:
            plt.bar(range(len(counts)), counts, label='Counts', alpha=alpha)
        if both or corrected:
            plt.bar(range(len(counts_corrected)), counts_corrected, label='Counts Corrected', alpha=alpha)

        num_qubits = self.get_num_qubits()

        state_labels = [''.join([x for x in p]) for p in product([0,1], repeat=num_qubits)]
        plt.xticks(range(len(state_labels)), state_labels)
        plt.xlabel('State')
        plt.ylabel('Counts')
        plt.title('Counts')

        if both:
            plt.legend()
        plt.show()

class PopulationShotsTimeSweepBase(PopulationShotsBase):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
    
        self.times = None

    def get_times(self):
        if self.times is None:
            self.acquire_data()
        return self.times

    def acquire_data(self):
        self.population_shots, self.counts, self.readout_qubits, self.confusion_matrices, self.times = acquire_data(self.filename, self.ramp)

        if self.counts is None:
            self.create_counts()

    def plot_counts(self, corrected=False, both=False):
        if both:
            raise ValueError("both cannot be True for PopulationShotsTimeSweepBase, use corrected instead.")
        
        if corrected:
            counts = self.get_counts_corrected()
        else:
            counts = self.get_counts()
        
        # Assume counts is a 2D array: [num_states, num_times]
        num_states, num_times = counts.shape

        times = self.get_times()
        
        # Create state labels for n-qubit system
        readout_indices = self.get_readout_indices()
        num_qubits = self.get_num_qubits()
        state_labels = [''.join([str(x) for x in p]) for p in product([0,1], repeat=num_qubits)]

        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate arrays
        x = np.arange(num_states)  # State index
        y = times  # Time index
        _x, _y = np.meshgrid(x, y, indexing='ij')
        
        # Flatten for bar3d
        x_flat = _x.flatten()
        y_flat = _y.flatten()
        z_flat = np.zeros_like(x_flat)
        dx = 0.8 * np.ones_like(x_flat)
        dy = 0.8 * np.ones_like(y_flat)
        
        # Flatten the counts data
        counts_flat = counts.flatten()

        print(f'shapes')
        print(counts.shape)
        print(_x.shape)
        print(_y.shape)

        print(f'flat')
        print(counts_flat.shape)
        print(x_flat.shape)
        print(y_flat.shape)
        print(z_flat.shape)


        # Create the 3D bar plot
        ax.bar3d(x_flat, y_flat, z_flat, dx, dy, counts_flat, shade=True)
        
        # Set labels and formatting
        ax.set_xlabel('State')
        ax.set_ylabel('Times')
        ax.set_zlabel('Counts')
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels[:num_states], rotation=45)
        
        title = 'Counts Over Time'
        if corrected:
            title += ' (Corrected)'
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()

    def plot_population(self, corrected=False, both=False, plot_sum=False, beamsplitter_time=None):
        population_average = self.get_population_average()
        population_sum = np.sum(population_average, axis=0)
        if both or corrected:
            population_corrected = self.get_population_corrected()
            population_sum_corrected = np.sum(population_average, axis=0)
        num_qubits, num_times = population_average.shape

        times = self.get_times()

        readout_qubits = self.get_readout_qubits()

        fig, axes = plt.subplots(num_qubits, 1, figsize=(12, 3 * num_qubits), sharex=True)

        if num_qubits == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if both or not corrected:
                ax.plot(times, population_average[i], label=f'Qubit {readout_qubits[i]}')
                if plot_sum:
                    ax.plot(times, population_sum, linestyle=':', color='gray', label='total population')
            if both or corrected:
                ax.plot(times, population_corrected[i], label=f'Qubit {readout_qubits[i]} Corrected')
                if plot_sum:
                    ax.plot(times, population_sum_corrected, linestyle=':', color='black', label='total population corrected')

            if both:
                ax.legend()

            ax.set_ylabel('Population')
            ax.set_title(f'Qubit {i+1} Population Over Time')
            ax.legend()
            ax.grid(True)

            if not beamsplitter_time is None:
                ax.axvline(beamsplitter_time, linestyle=':', color='red')

        axes[-1].set_xlabel('Time Index')
        plt.tight_layout()
        plt.show()

    def plot_population_difference(self, pairs, corrected=False, both=False, beamsplitter_time=None):
        population_average = self.get_population_average()
        population_differences = np.array([population_average[pair[1]] - population_average[pair[0]] for pair in pairs])
        if both or corrected:
            population_corrected = self.get_population_corrected()
            population_differences_corrected = np.array([population_corrected[pair[1]] - population_corrected[pair[0]] for pair in pairs])
        
        num_qubits, num_times = population_average.shape

        times = self.get_times()

        readout_qubits = self.get_readout_qubits()

        fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 3 * num_qubits), sharex=True)

        if num_qubits == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if both or not corrected:
                ax.plot(times, population_differences[i], label=f'Q{pairs[i][1]+1} - Q{pairs[i][0]+1}')
            if both or corrected:
                ax.plot(times, population_differences_corrected[i], label=f'Q{pairs[i][1]+1} - Q{pairs[i][0]+1} Corrected')

            if both:
                ax.legend()

            ax.set_ylabel('Population')
            ax.set_title(f'Q{pairs[i][1]+1} - Q{pairs[i][0]+1}')
            ax.legend()
            ax.grid(True)

            if not beamsplitter_time is None:
                ax.axvline(beamsplitter_time, linestyle=':', color='red')

        axes[-1].set_xlabel('Time Index')
        plt.tight_layout()
        plt.show()

    def plot_standard_deviation(self, corrected=False, both=False):
        standard_deviation = self.get_standard_deviation()
        if both or corrected:
            standard_deviation_corrected = self.get_standard_deviation_corrected()
        num_qubits, num_times = standard_deviation.shape

        times = self.get_times()
        readout_qubits = self.get_readout_qubits()

        fig, axes = plt.subplots(num_qubits, 1, figsize=(12, 3 * num_qubits), sharex=True)

        if num_qubits == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if both or not corrected:
                ax.plot(times, standard_deviation[i], label=f'Qubit {readout_qubits[i]} Std Dev')
            if both or corrected:
                ax.plot(times, standard_deviation_corrected[i], label=f'Qubit {readout_qubits[i]} Std Dev Corrected')

            if both:
                ax.legend()

            ax.set_ylabel('Standard Deviation')
            ax.set_title(f'Qubit {i+1} Standard Deviation Over Time')
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel('Time Index')
        plt.tight_layout()
        plt.show()
        
    def plot_covariance(self, corrected=False, both=False):
        covariance = self.get_covariance()
        if both or corrected:
            covariance_corrected = self.get_covariance_corrected()
        num_qubits, _, num_times = covariance.shape

        times = self.get_times()
        readout_qubits = self.get_readout_qubits()

        # Get all unique pairs i != j (i < j)
        pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        n_pairs = len(pairs)
        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
        axes = axes.flatten()

        for idx, (i, j) in enumerate(pairs):
            ax = axes[idx]
            label_base = f'Qubits {readout_qubits[i]}-{readout_qubits[j]}'
            if both or not corrected:
                ax.plot(times, covariance[i, j], label=label_base)
            if both or corrected:
                ax.plot(times, covariance_corrected[i,j], label=label_base + ' Corrected')
            if both:
                ax.legend()
            ax.set_ylabel('Covariance')
            ax.set_title(f'Covariance: Qubits {readout_qubits[i]} & {readout_qubits[j]}')
            ax.grid(True)

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            fig.delaxes(axes[idx])

        axes[0].set_xlabel('Time Index')
        plt.tight_layout()
        plt.show()

    def plot_covariance_sum(self, pair_1, pair_2, corrected=False, both=False):
        covariance = self.get_covariance()
        
        covariance_sum = 0
        covariance_sum += covariance[pair_1[0], pair_2[0]] + covariance[pair_1[1], pair_2[1]]
        covariance_sum -= covariance[pair_1[0], pair_2[1]] + covariance[pair_1[1], pair_2[0]]
    
        if both or corrected:
            covariance_corrected = self.get_covariance_corrected()
            
            covariance_sum_corrected = 0
            covariance_sum_corrected += covariance_corrected[pair_1[0], pair_2[0]] + covariance_corrected[pair_1[1], pair_2[1]]
            covariance_sum_corrected -= covariance_corrected[pair_1[0], pair_2[1]] + covariance_corrected[pair_1[1], pair_2[0]]

        times = self.get_times()
        readout_qubits = self.get_readout_qubits()

        fig, ax = plt.subplots(figsize=(12, 6))

        if both or not corrected:
            ax.plot(times, covariance_sum, 'o-', label=f'Covariance Sum')
        if both or corrected:
            ax.plot(times, covariance_sum_corrected, 'o-', label=f'Covariance Sum Corrected')

        if both:
            ax.legend()

        ax.set_ylabel('Covariance Sum')
        ax.set_title(f'Covariance Sum Over Time for Pairs ({pair_1}) and ({pair_2})')
        ax.grid(True)
        ax.set_xlabel('Time Index')

        plt.tight_layout()
        plt.show()
        

class RampPopulationShotsMeasurement(PopulationShotsBase):
    ramp = True


class RampOscillationShotsMeasurement(PopulationShotsTimeSweepBase):
    ramp = True

def reduce_counts(counts, index_1, index_2, num_qubits):
    """
    Reduce a counts matrix to 2D by summing over all indices except index_1 and index_2.

    Parameters:
    - counts_matrix: 4D array with shape [2, 2, 2, 2] for 4 qubits
    - index_1, index_2: The two indices to keep (0-based)
    - num_qubits: Total number of qubits (should be 4 for your case)
    
    Returns:
    - reduced_matrix: 2x2 matrix with counts for qubits index_1 and index_2
    """
    # Get all indices except index_1 and index_2
    other_indices = [idx for idx in range(num_qubits) if idx not in [index_1, index_2]]

    # Reshape counts to (2, 2, ..., 2) for num_qubits
    counts_reshaped = counts.reshape([2] * num_qubits + list(counts.shape[1:]), order='C')

    # Sum over all other indices
    reduced_matrix = np.sum(counts_reshaped, axis=tuple(other_indices))

    # If index_1 > index_2, we need to transpose to maintain order
    if index_1 > index_2:
        reduced_matrix = reduced_matrix.T


    # Flatten the reduced_counts_matrix so that [0,0], [0,1], [1,0], [1,1] are the four entries of a 4x1 vector
    reduced_vector = reduced_matrix.reshape([4] + list(counts.shape[1:]), order='C')

    return reduced_vector

def acquire_data(filepath, ramp=False):

    with h5py.File(filepath, "r") as f:
        
        time_units = 2.32515 / 16 # tproc_V1
        time_units = 2.32515*2 / 16 # tproc_V2  

        # for i in f:
            # print(f'{i}: {f[i][()]}')
            # print(i)

        

        population = f['population_shots'][()]
        confusion_matrix = f['confusion_matrix'][()]

        counts = None
        if 'counts' in f:
            counts = f['counts'][()]



        times = None
        if 'expt_samples' in f:
            times = f['expt_samples'][()]
        else:
            if ramp:
                times = f['expt_samples2'][()]


        readout_list = [int(i) for i in f['readout_list'][()]]

    times *= time_units

    if times is not None:
        return population, counts, readout_list, confusion_matrix, times
    else:
        return population, counts, readout_list, confusion_matrix


def generate_ramp_population_shots_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampPopulationShots\RampPopulationShots_{}\RampPopulationShots_{}_{}_data.h5'.format(date_code, date_code, time_code)


def generate_population_shots_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\PopulationShots\PopulationShots_{}\PopulationShots_{}_{}_data.h5'.format(date_code, date_code, time_code)

def generate_oscillation_population_shots_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\OscillationPopulationShots\OscillationPopulationShots_{}\OscillationPopulationShots_{}_{}_data.h5'.format(date_code, date_code, time_code)



def generate_ramp_oscillation_population_shots_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampOscillationPopulationShots\RampOscillationPopulationShots_{}\RampOscillationPopulationShots_{}_{}_data.h5'.format(date_code, date_code, time_code)




def generate_ramp_beamsplitter_correlations_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampBeamsplitterCorrelationsR\RampBeamsplitterCorrelationsR_{}\RampBeamsplitterCorrelationsR_{}_{}_data.h5'.format(date_code, date_code, time_code)


def generate_ramp_double_jump_correlations_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampDoubleJumpCurrentCorrelations\RampDoubleJumpCurrentCorrelations_{}\RampDoubleJumpCurrentCorrelations_{}_{}_data.h5'.format(date_code, date_code, time_code)
