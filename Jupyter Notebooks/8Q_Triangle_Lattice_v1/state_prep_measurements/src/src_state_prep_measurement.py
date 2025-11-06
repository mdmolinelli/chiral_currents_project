from itertools import product
import h5py
import matplotlib.pyplot as plt
import numpy as np



class StatePrepMeasurement:

    # time_units = 2.32515 / 16 # tproc_V1
    time_units = 2.32515*2 / 16 # tproc_V2

    def __init__(self, filename, num_particles):
        self.filename = filename
        self.num_particles = num_particles

        self.counts = None
        self.counts_post_selected = None

        self.probabilities = None
        self.probabilities_post_selected = None

        self.bitstrings = None
        self.bitstrings_post_selected = None

        self.readout_list = None
        self.num_qubits = None

        self.times = None

    def acquire_data(self):
        self.counts, self.times, self.readout_list = acquire_data(self.filename)

    def get_times(self):
        if self.times is None:
            self.acquire_data()
        return self.times

    def get_counts(self):
        if self.counts is None:
            self.acquire_data()
        return self.counts
    
    def get_counts_post_selected(self):
        if self.counts_post_selected is None:
            self.post_select_counts()
        return self.counts_post_selected
    
    def get_probabilities(self):
        if self.probabilities is None:
            counts = self.get_counts()
            self.probabilities = counts / np.sum(counts, axis=0)
        return self.probabilities
    
    def get_probabilities_post_selected(self):
        if self.probabilities_post_selected is None:
            counts_post_selected = self.get_counts_post_selected()
            self.probabilities_post_selected = counts_post_selected / np.sum(counts_post_selected, axis=0)
        return self.probabilities_post_selected

    def get_readout_list(self):
        if self.readout_list is None:
            self.acquire_data()
        return self.readout_list
    
    def get_num_qubits(self):
        if self.num_qubits is None:
            self.num_qubits = len(self.get_readout_list())
        
        return self.num_qubits
    
    def get_bitstrings(self):
        if self.bitstrings is None:
            self.bitstrings = self.generate_bitstrings()
        return self.bitstrings
    
    def get_bitstrings_post_selected(self):
        if self.bitstrings_post_selected is None:
            self.post_select_counts()
        return self.bitstrings_post_selected
    
    def post_select_counts(self):
        counts = self.get_counts()
        num_qubits = self.get_num_qubits()

        bitstrings = self.generate_bitstrings()
        selected_indices = [i for i in range(len(bitstrings)) if sum(bitstrings[i]) == self.num_particles]

        # print(f'Number of basis states before post-selection: {len(bitstrings)}')
        # print(f'Number of basis states after post-selection: {len(selected_indices)}')

        self.bitstrings_post_selected = [bitstrings[i] for i in selected_indices]
        self.counts_post_selected = counts[selected_indices, :]

    def plot_state_ns(self, time):
        index = self.ns_to_index(time)
        self.plot_state_index(index)

    def plot_state_samples(self, samples):
        index = self.samples_to_index(samples)
        self.plot_state_index(index)

    def plot_state_index(self, index):

        counts = self.get_counts()

        counts_index = counts[:,index]
        probs = counts_index / np.sum(counts_index)

        

        k = 20
        bitstrings = self.generate_bitstrings()

        idx_sorted = np.argsort(probs)[::-1]
        top_idx = idx_sorted[:k]
        top_probs = probs[top_idx]

        top_labels = []
        for i in range(len(top_idx)):
            label = ''.join(map(str, bitstrings[top_idx[i]]))
            top_labels.append(label)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(range(k), top_probs)
        ax.set_xticks(range(k))
        ax.set_xticklabels(top_labels, rotation=70, fontsize=9)
        ax.set_ylabel('Probability')
        ax.set_title(f'Top {k} outcomes by probability')
        plt.tight_layout()
        plt.show()

    def plot_states(self):
        # counts = self.get_counts()
        # times = self.get_times()
        # bitstrings = self.generate_bitstrings()
        # num_bitstrings = len(bitstrings)
        # num_times = len(times)

        # fig = plt.figure(figsize=(16, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # _x = np.arange(num_times)
        # _y = np.arange(num_bitstrings)
        # _xx, _yy = np.meshgrid(_x, _y)
        # x, y = _xx.ravel(), _yy.ravel()

        # z = np.zeros_like(x)
        # dz = counts.ravel(order='F')  # shape: (num_bitstrings, num_times)

        # ax.bar3d(x, y, z, 1, 1, dz, shade=True)

        # ax.set_xlabel('Times (ns)')
        # ax.set_ylabel('Bitstring')
        # ax.set_zlabel('Counts')

        # ax.set_xticks(np.arange(num_times)[::max(1, num_times // 10)])
        # ax.set_xticklabels([f"{t:.1f}" for t in times[::max(1, num_times // 10)]], rotation=45)

        # ax.set_yticks(np.arange(num_bitstrings)[::max(1, num_bitstrings // 20)])
        # ax.set_yticklabels([''.join(map(str, bitstrings[i])) for i in range(0, num_bitstrings, max(1, num_bitstrings // 20))])

        # plt.tight_layout()
        # plt.show()

        self.plot_states_subset()

    def plot_states_reference(self, reference_time=0, subset_size=20):

        reference_index = self.ns_to_index(reference_time)
        counts = self.get_counts()

        bitstrings = self.generate_bitstrings()
        counts_at_ref = counts[:, reference_index]
        top_indices = np.argsort(counts_at_ref)[::-1][:subset_size]
        subset_bitstrings = [bitstrings[i] for i in top_indices][::-1]
        self.plot_states_subset(subset_bitstrings=subset_bitstrings)


    def plot_states_subset(self, subset_bitstrings=None):


        if subset_bitstrings is None:
            subset_bitstrings = bitstrings
            
        subset_size = len(subset_bitstrings)

        for i in range(subset_size):
            if isinstance(subset_bitstrings[i], str):
                subset_bitstrings[i] = [int(b) for b in subset_bitstrings[i]]

        counts = self.get_counts()
        times = self.get_times()
        bitstrings = self.generate_bitstrings()

        
        num_times = len(times)

        # Map bitstrings to their indices
        bitstring_to_index = {tuple(bs): i for i, bs in enumerate(bitstrings)}


        subset_indices = [bitstring_to_index[tuple(bs)] for bs in subset_bitstrings if tuple(bs) in bitstring_to_index]
        subset_labels = [''.join(map(str, bitstrings[i])) for i in subset_indices]

        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111, projection='3d')

        _x = np.arange(num_times)
        _y = np.arange(subset_size)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        z = np.zeros_like(x)
        dz = np.array([counts[subset_indices[j], i] for j in range(subset_size) for i in range(num_times)])

        ax.bar3d(x, y, z, 1, 1, dz, shade=True)

        ax.set_xlabel('Times (ns)')
        ax.set_ylabel('Bitstring')
        ax.set_zlabel('Counts')

        ax.set_xticks(np.arange(num_times)[::max(1, num_times // 10)])
        ax.set_xticklabels([f"{t:.1f}" for t in times[::max(1, num_times // 10)]], rotation=45)

        if subset_size <= 20:
            ax.set_yticks(np.arange(subset_size))
            ax.set_yticklabels(subset_labels)
        else:
            ax.set_yticks(np.arange(subset_size)[::max(1, subset_size // 20)])
            ax.set_yticklabels([''.join(map(str, bitstrings[i])) for i in range(0, subset_size, max(1, subset_size // 20))])


        plt.tight_layout()
        plt.show()



    def ns_to_index(self, ns):
        times = self.get_times()

        index = int(np.argmin(np.abs(times - ns)))
        return index

    def samples_to_index(self, samples):
        return self.ns_to_index(self.samples_to_ns(samples))

    def samples_to_ns(self, samples):
        return samples * self.time_units

    def generate_bitstrings(self):
        num_qubits = self.get_num_qubits()
        bitstrings = list(product([0, 1], repeat=num_qubits))
        return bitstrings



def acquire_data(filepath):

    time_units = 2.32515 / 16 # tproc_V1
    time_units = 2.32515*2 / 16 # tproc_V2

    with h5py.File(filepath, "r") as f:
        
        # for i in f:
            # print(f'{i}: {f[i][()]}')
            # print(i)

        counts = f['counts'][()]
        times = f['expt_samples'][()]

        readout_list = [int(i) for i in f['readout_list'][()]]

    times *= time_units

    return counts, times, readout_list

def generate_state_prep_filename(year, month, day, hour, minute, second):
    date_code = f'{year}_{month}_{day}'
    time_code = f'{hour}_{minute}_{second}'
    return r'V:\QSimMeasurements\Measurements\8QV1_Triangle_Lattice\TimeVsPopulationShots\TimeVsPopulationShots_{}\TimeVsPopulationShots_{}_{}_data.h5'.format(date_code, date_code, time_code)

