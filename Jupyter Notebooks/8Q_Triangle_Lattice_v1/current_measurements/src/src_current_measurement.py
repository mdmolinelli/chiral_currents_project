from itertools import count
import h5py
import matplotlib.pyplot as plt
import numpy as np


class CurrentMeasurementCalibration:
    def __init__(self, filename, readout_pair_1, readout_pair_2, singleshot_measurements=None, time_offset=None, post_select=False, particle_number=0):
        
        '''
        :param filename: Path to the data file containing measurement results.
        :param readout_pair_1: First pair of readout indices.
        :param readout_pair_2: Second pair of readout indices.
        :param singleshot_measurements: Dictionary containing single-shot one-qubit and two-qubit measurement results
            with ints or tuples of readout indices as keys. Defaults to None.
        :param time_offset: Time offset to apply to the second set of readout indices in units of number of samples. Defaults to 0.

        '''
        self.filename = filename
        self.readout_pair_1 = readout_pair_1
        self.readout_pair_2 = readout_pair_2
        self.readout_pairs = [readout_pair_1, readout_pair_2]

        self.singleshot_measurements = singleshot_measurements

        if time_offset is None:
            time_offset = [0, 0, 0, 0]
        time_offset = np.array(time_offset) - np.min(time_offset)  # Ensure time_offset is non-negative
        self.time_offset = time_offset


        self.post_select = post_select
        self.particle_number = particle_number

        if post_select:
            print(f'post selecting on particle number = {particle_number}')

        self.times = None
        self.population_shots = None
        self.population_average = None
        self.population_corrected = None

        self.population_difference_shots = None
        self.population_difference_average = None
        self.population_difference_corrected = None

        # these define the four terms of the correlator <(n2-n1)(n4-n3)> = <n1n3> + <n2n4> - <n1n4> - <n2n3> 
        self.n1n3_shots = None
        self.n2n4_shots = None
        self.n1n4_shots = None
        self.n2n3_shots = None

        self.n1n3_average = None
        self.n2n4_average = None
        self.n1n4_average = None
        self.n2n3_average = None

        self.n1n3_average_corrected = None
        self.n2n4_average_corrected = None
        self.n1n4_average_corrected = None
        self.n2n3_average_corrected = None


        self.counts_matrix = None

        self.correlation_shots = None
        self.correlation_average = None
        self.correlation_average_corrected = None

        self.n_terms_corrected = {}

    def acquire_data(self):

        data = acquire_data(self.filename)

        if len(data) == 3:
            self.times, self.population_shots, self.nn_correlations = data
        else:
            self.times, self.population_shots = data

        print(self.population_shots.shape)


        if not np.all(self.time_offset == 0):

            largest_offset = np.max(self.time_offset)
            new_length = self.population_shots.shape[1] - abs(largest_offset)

            population_shots_offset = np.zeros((self.population_shots.shape[0], new_length, self.population_shots.shape[2]), dtype=self.population_shots.dtype)


            for i in range(len(self.time_offset)):
                channel_offset = self.time_offset[i]
                population_shots_offset[i] = self.population_shots[i, channel_offset:channel_offset + new_length]

            self.population_shots = population_shots_offset

            # modify time array too
            timestep = self.times[1] - self.times[0]
            new_length = self.population_shots.shape[1]
            self.times = np.arange(new_length) * timestep

            

        print('after padding')
        print(self.population_shots.shape)


    def get_times(self):
        if self.times is None:
            self.acquire_data()
        return self.times

    def get_population_shots(self):
        if self.population_shots is None:
            self.acquire_data()
        return self.population_shots
    
    def get_population_average(self):
        if self.population_average is None:
            population_shots = self.get_population_shots()
            self.population_average = np.zeros_like(np.mean(population_shots, axis=-1))
            if self.post_select:
                # If post-selecting, we only consider the shots that match the particle number
                for i in range(population_shots.shape[1]):
                    count = 0
                    for j in range(population_shots.shape[2]):
                        if np.sum(population_shots[:, i, j]) == self.particle_number:
                            self.population_average[:, i] += population_shots[:, i, j]
                            count += 1
                    self.population_average[:, i] /= count
            else:
                self.population_average = np.mean(population_shots, axis=-1)
        return self.population_average
    
    def get_population_corrected(self):
        if self.population_corrected is None:
            self.correct_populations()
        return self.population_corrected
    
    def get_population_difference_shots(self):
        if self.population_difference_shots is None:
            population_shots = self.get_population_shots()
            population_difference_shots = np.zeros((2, population_shots.shape[1], population_shots.shape[2]), dtype=population_shots.dtype)
            population_difference_shots[0,:,:] = population_shots[self.readout_pair_1[1], :, :] - population_shots[self.readout_pair_1[0], :, :]
            population_difference_shots[1,:,:] = population_shots[self.readout_pair_2[1], :, :] - population_shots[self.readout_pair_2[0], :, :]
            self.population_difference_shots = population_difference_shots
        return self.population_difference_shots
    
    def get_population_difference_average(self):
        if self.population_difference_average is None:
            population_difference_shots = self.get_population_difference_shots()
            self.population_difference_average = np.mean(population_difference_shots, axis=-1)
        return self.population_difference_average
    
    def get_population_difference_corrected(self):  
        if self.population_difference_corrected is None:
            population_corrected = self.get_population_corrected()
            population_difference_corrected = np.zeros((2, population_corrected.shape[1]), dtype=population_corrected.dtype)
            population_difference_corrected[0,:] = population_corrected[self.readout_pair_1[1], :] - population_corrected[self.readout_pair_1[0], :]
            population_difference_corrected[1,:] = population_corrected[self.readout_pair_2[1], :] - population_corrected[self.readout_pair_2[0], :]
            self.population_difference_corrected = population_difference_corrected
        return self.population_difference_corrected
    
    def get_n1n3_shots(self):
        if self.n1n3_shots is None:
            population_shots = self.get_population_shots()
            self.n1n3_shots = population_shots[self.readout_pair_1[0], :, :] * population_shots[self.readout_pair_2[0], :, :]
        return self.n1n3_shots

    def get_n2n4_shots(self):
        if self.n2n4_shots is None:
            population_shots = self.get_population_shots()
            self.n2n4_shots = population_shots[self.readout_pair_1[1], :, :] * population_shots[self.readout_pair_2[1], :, :]
        return self.n2n4_shots

    def get_n1n4_shots(self):
        if self.n1n4_shots is None:
            population_shots = self.get_population_shots()
            self.n1n4_shots = population_shots[self.readout_pair_1[0], :, :] * population_shots[self.readout_pair_2[1], :, :]
        return self.n1n4_shots

    def get_n2n3_shots(self):
        if self.n2n3_shots is None:
            population_shots = self.get_population_shots()
            self.n2n3_shots = population_shots[self.readout_pair_1[1], :, :] * population_shots[self.readout_pair_2[0], :, :]
        return self.n2n3_shots
    
    def get_n1n3_average(self, corrected=False):
        if corrected:
            return self.get_n1n3_average_corrected()
        if self.n1n3_average is None:
            n1n3_shots = self.get_n1n3_shots()
            self.n1n3_average = np.mean(n1n3_shots, axis=-1)
        return self.n1n3_average

    def get_n2n4_average(self, corrected=False):
        if corrected:
            return self.get_n2n4_average_corrected()
        if self.n2n4_average is None:
            n2n4_shots = self.get_n2n4_shots()
            self.n2n4_average = np.mean(n2n4_shots, axis=-1)
        return self.n2n4_average

    def get_n1n4_average(self, corrected=False):
        if corrected:
            return self.get_n1n4_average_corrected()
        if self.n1n4_average is None:
            n1n4_shots = self.get_n1n4_shots()
            self.n1n4_average = np.mean(n1n4_shots, axis=-1)
        return self.n1n4_average

    def get_n2n3_average(self, corrected=False):
        if corrected:
            return self.get_n2n3_average_corrected()
        if self.n2n3_average is None:
            n2n3_shots = self.get_n2n3_shots()
            self.n2n3_average = np.mean(n2n3_shots, axis=-1)
        return self.n2n3_average
    
    def get_n_term(self, index_1, index_2):
        if index_1 == 0:
            if index_2 == 2:
                return self.get_n1n3_average()
            elif index_2 == 3:
                return self.get_n1n4_average()
        elif index_1 == 1:
            if index_2 == 2:
                return self.get_n2n3_average()
            elif index_2 == 3:
                return self.get_n2n4_average()

    def get_n1n3_average_corrected(self):
        if self.n1n3_average_corrected is None:
            self.n1n3_average_corrected = self.get_n_term_corrected(0,2)
        return self.n1n3_average_corrected
    
    def get_n1n4_average_corrected(self):
        if self.n1n4_average_corrected is None:
            self.n1n4_average_corrected = self.get_n_term_corrected(0,3)
        return self.n1n4_average_corrected
    
    def get_n2n3_average_corrected(self):
        if self.n2n3_average_corrected is None:
            self.n2n3_average_corrected = self.get_n_term_corrected(1,2)
        return self.n2n3_average_corrected
    
    def get_n2n4_average_corrected(self):
        if self.n2n4_average_corrected is None:
            self.n2n4_average_corrected = self.get_n_term_corrected(1,3)
        return self.n2n4_average_corrected
    
    def get_n_term_corrected(self, index_1, index_2):
        key = (index_1, index_2)
        if not key in self.n_terms_corrected:
            self.correct_n_terms(index_1, index_2)
        return self.n_terms_corrected[key]

    def get_counts_matrix(self):
        if self.counts_matrix is None:
            self.construct_counts_matrix()
        return self.counts_matrix

    def get_correlation_shots(self):
        if self.correlation_shots is None:
            self.correlation_shots = self.get_population_difference_shots()[0,:,:] * self.get_population_difference_shots()[1,:,:]
        return self.correlation_shots
    
    def get_correlation_average(self, corrected=False):
        if corrected:
            return self.get_correlation_average_corrected()
        if self.correlation_average is None:
            correlation_shots = self.get_correlation_shots()
            self.correlation_average = np.mean(correlation_shots, axis=-1)
        return self.correlation_average
    
    def get_correlation_average_corrected(self):
        if self.correlation_average_corrected is None:
            n1n3_average_corrected = self.get_n1n3_average_corrected()
            n2n4_average_corrected = self.get_n2n4_average_corrected()
            n1n4_average_corrected = self.get_n1n4_average_corrected()
            n2n3_average_corrected = self.get_n2n3_average_corrected()

            self.correlation_average_corrected = n1n3_average_corrected + n2n4_average_corrected - n1n4_average_corrected - n2n3_average_corrected
        return self.correlation_average_corrected

    def construct_counts_matrix(self):

        population_shots = self.get_population_shots()


        self.counts_matrix = np.zeros((2, 2, 2, 2, population_shots.shape[1]), dtype=population_shots.dtype)
        # For each time step, count the number of shots for each bitstring ijkl
        for t in range(population_shots.shape[1]):
            # For each shot at time t
            for shot in range(population_shots.shape[2]):
                # Get the bitstring for the four qubits at this time and shot
                bits = (
                    int(population_shots[self.readout_pair_1[0], t, shot]),
                    int(population_shots[self.readout_pair_1[1], t, shot]),
                    int(population_shots[self.readout_pair_2[0], t, shot]),
                    int(population_shots[self.readout_pair_2[1], t, shot])
                )

                if self.post_select:
                    if sum(bits) == self.particle_number:
                        self.counts_matrix[bits[0], bits[1], bits[2], bits[3], t] += 1
                else:
                    self.counts_matrix[bits[0], bits[1], bits[2], bits[3], t] += 1

    def correct_populations(self):
        if self.singleshot_measurements is None:
            raise RuntimeError('Need to provide singleshot two-qubit measurements for before correcting n terms')

        populations = self.get_population_average()
        self.population_corrected = np.zeros_like(populations)

        for i in range(len(populations)):
            readout_index = self.readout_pairs[i//2][i%2]

            confusion_matrix = self.singleshot_measurements[readout_index].get_confusion_matrix()
            inverse_confusion_matrix = np.linalg.inv(confusion_matrix)

            measured_population_vector = np.array([1-populations[i, :], populations[i, :]])

            real_population_vector = inverse_confusion_matrix @ measured_population_vector
            self.population_corrected[i,:] = real_population_vector[-1,:]


    def correct_n_terms(self, index_1, index_2):

        print(f'correcting for index {index_1} and index {index_2}')

        if self.singleshot_measurements is None:
            raise RuntimeError('Need to provide singleshot two-qubit measurements for before correcting n terms')

        counts_matrix = self.get_counts_matrix()

        # Trace out (sum over) the irrelevant indices to get the reduced counts for (index_1, index_2)
        # counts_matrix shape: [2, 2, 2, 2, n_times]
        # We want to sum over the other two indices (besides index_1 and index_2)
        all_indices = [0, 1, 2, 3]
        irrelevant_indices = [idx for idx in all_indices if idx not in [index_1, index_2]]

        counts_reduced = np.sum(counts_matrix, axis=tuple(irrelevant_indices))

        # Convert counts_reduced to shape (4, n_times) with order: 00, 01, 10, 11
        # counts_reduced shape is (2, 2, n_times), so we flatten the first two axes
        counts_reduced_reshaped = counts_reduced.reshape(4, counts_reduced.shape[-1])

        confusion_matrix = self.singleshot_measurements[(index_1, index_2)].get_confusion_matrix()
        confusion_matrix_inverse = np.linalg.inv(confusion_matrix)

        counts_reduced_reshaped_corrected = confusion_matrix_inverse @ counts_reduced_reshaped
        populations_corrected  = counts_reduced_reshaped_corrected / np.sum(counts_reduced_reshaped, axis=0)

        n_term_corrected = populations_corrected[-1,:]

        self.n_terms_corrected[(index_1, index_2)] = n_term_corrected

    def plot_population_average(self, corrected=False, both=False, plot_sum=False):
        times = self.get_times()
        population_average = self.get_population_average()
        if corrected or both:
            population_corrected = self.get_population_corrected()
        
        fig, axes = plt.subplots(population_average.shape[0], 1, figsize=(8, 2 * population_average.shape[0]), sharex=True)
        for i in range(population_average.shape[0]):
            if both:
                axes[i].plot(times, population_average[i, :], label='Population Average')
                axes[i].plot(times, population_corrected[i, :], label='Population Corrected')
                axes[i].legend()
            else:
                if corrected:
                    axes[i].plot(times, population_corrected[i, :])
                else:
                    axes[i].plot(times, population_average[i, :])

            if plot_sum:
                if not corrected or both:
                    axes[i].plot(times, np.sum(population_average, axis=0), label='Total Population', linestyle='--')
                    axes[i].legend()
                if corrected or both:
                    axes[i].plot(times, np.sum(population_corrected, axis=0), label='Total Population Corrected', linestyle='--')
                    axes[i].legend()

            axes[i].set_ylabel(f'Qubit {i+1}')
            axes[i].set_title(f'Population Average for Qubit {i+1}')
        axes[-1].set_xlabel('Time (ns)')
        plt.tight_layout()
        plt.show()

    def plot_population_shots(self, shot_index=0):
        times = self.get_times()
        population_shots = self.get_population_shots()

        fig, axes = plt.subplots(population_shots.shape[0], 1, figsize=(8, 2 * population_shots.shape[0]), sharex=True)
        for i in range(population_shots.shape[0]):
            axes[i].plot(times, population_shots[i, :, shot_index])
            axes[i].set_ylabel(f'Qubit {i+1}')
            axes[i].set_title(f'Population Shot #{shot_index} for Qubit {i+1}')
        axes[-1].set_xlabel('Time (ns)')
        plt.tight_layout()
        plt.show()

    def plot_population_difference_shots(self, shot_index=0):
        times = self.get_times()
        population_difference_shots = self.get_population_difference_shots()

        fig, axes = plt.subplots(population_difference_shots.shape[0], 1, figsize=(8, 2 * population_difference_shots.shape[0]), sharex=True)
        for i in range(population_difference_shots.shape[0]):
            axes[i].plot(times, population_difference_shots[i, :, shot_index])
            axes[i].set_ylabel(f'Q{self.readout_pairs[i][1]+1} - Q{self.readout_pairs[i][0]+1}')
            axes[i].set_title(f'Population Difference Shot #{shot_index} for Qubits {self.readout_pairs[i][1]+1} and {self.readout_pairs[i][0]+1}')
        axes[-1].set_xlabel('Time (ns)')
        plt.tight_layout()
        plt.show()

    def plot_population_difference_average(self, corrected=False, both=False, beamsplitter_time=None):
        times = self.get_times()
        population_difference_average = self.get_population_difference_average()
        population_difference_corrected = self.get_population_difference_corrected()

        fig, axes = plt.subplots(population_difference_average.shape[0], 1, figsize=(8, 2 * population_difference_average.shape[0]), sharex=True)
        for i in range(population_difference_average.shape[0]):

            if not corrected or both:
                axes[i].plot(times, population_difference_average[i, :], label='Population Difference')
            
            if corrected or both:
                axes[i].plot(times, population_difference_corrected[i, :], label='Population Difference Corrected')

            if both:
                axes[i].legend()

            axes[i].set_ylabel(f'Q{self.readout_pairs[i][1]+1} - Q{self.readout_pairs[i][0]+1}')
            axes[i].set_title(f'Population Difference Average for Qubits {self.readout_pairs[i][1]+1} and {self.readout_pairs[i][0]+1}')

            if beamsplitter_time is not None:
                axes[i].axvline(beamsplitter_time, color='red', linestyle='--', label='Beamsplitter Time')
                axes[i].legend()

        axes[-1].set_xlabel('Time (ns)')
        plt.tight_layout()
        plt.show()

    def plot_counts_histogram_slice(self, time_index=0):
        counts_matrix = self.get_counts_matrix()
        bitstrings = [f"{i:04b}" for i in range(16)]
        counts = np.zeros(16, dtype=int)
        for idx, bitstr in enumerate(bitstrings):
            i, j, k, l = [int(b) for b in bitstr]
            counts[idx] = counts_matrix[i, j, k, l, time_index]

        plt.figure(figsize=(10, 6))
        plt.bar(bitstrings, counts)
        plt.xlabel("Bitstring")
        plt.ylabel("Counts")
        plt.title(f"Counts Histogram at Time Index {time_index}")
        plt.tight_layout()
        plt.show()

    def plot_counts_histogram(self):
        counts_matrix = self.get_counts_matrix()
        num_times = counts_matrix.shape[-1]
        bitstrings = [f"{i:04b}" for i in range(16)]

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = np.arange(len(bitstrings))
        y = np.arange(num_times)
        _x, _y = np.meshgrid(x, y, indexing='ij')

        # Flatten for bar3d
        x_flat = _x.flatten()
        y_flat = _y.flatten()
        z_flat = np.zeros_like(x_flat)
        dx = 0.8 * np.ones_like(x_flat)
        dy = 0.8 * np.ones_like(y_flat)
        counts = np.zeros_like(x_flat)

        for idx, bitstr in enumerate(bitstrings):
            for t in range(num_times):
                counts[idx * num_times + t] = counts_matrix[int(bitstr[0]), int(bitstr[1]), int(bitstr[2]), int(bitstr[3]), t]

        ax.bar3d(x_flat, y_flat, z_flat, dx, dy, counts, shade=True)
        ax.set_xlabel('Bitstring')
        ax.set_ylabel('Time Index')
        ax.set_zlabel('Counts')
        ax.set_xticks(x)
        ax.set_xticklabels(bitstrings, rotation=90)
        ax.set_title('Counts Histogram Over Time')
        plt.tight_layout()
        plt.show()

    def plot_correlation_average(self, corrected=False, both=False, beamsplitter_time=None):
        times = self.get_times()

        
        correlation_average = self.get_correlation_average(corrected=corrected)

        plt.figure(figsize=(10, 6))

        title = 'Correlation Average Over Time'
        if both:
            plt.plot(times, self.get_correlation_average(corrected=False), label='Correlation Average')
            plt.plot(times, self.get_correlation_average(corrected=True), label='Correlation Average (corrected)')
            plt.legend()
        else:
            plt.plot(times, correlation_average)

            if corrected:
                title += ' (corrected)'

        if beamsplitter_time is not None:
            plt.axvline(beamsplitter_time, color='red', linestyle='--', label='Beamsplitter Time')
            if not both:
                plt.legend()
        
        plt.xlabel('Time (ns)')
        plt.ylabel('Correlation Average')
        

        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_n_terms(self, corrected=False, both=False, beamsplitter_time=None):
        times = self.get_times()

        n1n3 = self.get_n1n3_average(corrected=corrected)
        n1n4 = self.get_n1n4_average(corrected=corrected)
        n2n3 = self.get_n2n3_average(corrected=corrected)
        n2n4 = self.get_n2n4_average(corrected=corrected)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        labels = [
            '$n_1n_3$',
            '$n_1n_4$',
            '$n_2n_3$',
            '$n_2n_4$'
        ]
        data = [n1n3, n1n4, n2n3, n2n4]

        if both:
            data_corrected = self.get_n1n3_average(corrected=True), self.get_n1n4_average(corrected=True), self.get_n2n3_average(corrected=True), self.get_n2n4_average(corrected=True)
            data_uncorrected = self.get_n1n3_average(corrected=False), self.get_n1n4_average(corrected=False), self.get_n2n3_average(corrected=False), self.get_n2n4_average(corrected=False)

        for i, ax in enumerate(axes.flat):
            ax.set_title(labels[i])
            ax.set_ylabel('Value')
            ax.grid()
            if both:
                ax.plot(times, data_uncorrected[i], label=labels[i])
                ax.plot(times, data_corrected[i], linestyle='--', label=labels[i] + ' (corrected)')
                ax.legend()
            else:
                ax.plot(times, data[i], label=labels[i])

            if beamsplitter_time is not None:
                ax.axvline(beamsplitter_time, color='red', linestyle='--', label='Beamsplitter Time')
                if not both:
                    ax.legend()

            ax.set_title(labels[i])
        
        axes[1, 0].set_xlabel('Time (ns)')
        axes[1, 1].set_xlabel('Time (ns)')
        
        axes[0, 0].set_ylabel('Value')
        axes[1, 0].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

def acquire_data(filepath):
    with h5py.File(filepath, "r") as f:
        
        time_units = 2.32515 / 16 # tproc_V1
        time_units = 2.32515*2 / 16 # tproc_V2
        
        for i in f:
            # print(f'{i}: {f[i][()]}')
            print(i)


        try:
            times = f['expt_cycles2'][()]
        except:
            try:
                times = f['expt_samples2'][()]
            except:
                times = f['expt_samples'][()]


        nn_correlations = None
        if 'nn_correlations' in f:
            nn_correlations = f['nn_correlations'][()]

        population_shots = f['population_shots'][()]

        print(f'timestep before multiplication: {times[1] - times[0]} ns')

    times *= time_units
    print(f'timestep after multiplication: {times[1] - times[0]} ns')

    if nn_correlations is not None:
        return times, population_shots, nn_correlations
    else:
        return times, population_shots


def generate_current_calibration_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\CurrentCalibration_1D_Shots\CurrentCalibration_1D_Shots_{}\CurrentCalibration_1D_Shots_{}_{}_data.h5'.format(date_code, date_code, time_code)



def generate_ramp_beamsplitter_correlations_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampBeamsplitterCorrelationsR\RampBeamsplitterCorrelationsR_{}\RampBeamsplitterCorrelationsR_{}_{}_data.h5'.format(date_code, date_code, time_code)


def generate_ramp_beamsplitter_correlations_clean_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\RampBeamsplitterCleanTiming\RampBeamsplitterCleanTiming_{}\RampBeamsplitterCleanTiming_{}_{}_data.h5'.format(date_code, date_code, time_code)
